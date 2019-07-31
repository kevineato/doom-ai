from collections import deque
import cv2 as cv
import numpy as np
from pandas import get_dummies
import pickle
import random
import tensorflow as tf
from time import sleep
from tqdm import trange
from vizdoom import *

def process_frame(frame, new_size):
    """Helper function to resize and grayscale a given frame."""
    # change frame from (depth, width, height) to (width, height, depth)
    frame = np.rollaxis(frame, 0, 3)

    # resize to new_size and grayscale the frame
    frame = cv.resize(frame, new_size, interpolation=cv.INTER_AREA)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    return frame

def dqn(in_shape, num_actions, learn_rate):
    """Builds a single deep q network model using keras.

    Args:
        in_shape: The shape of the input to the network.
        num_actions: Number of allowed actions for agent to make.
        learn_rate: Learning rate for training of the model.

    Returns:
        A keras Sequential model compiled using the Nadam optimizer
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,
                                     activation=tf.keras.activations.relu,
                                     input_shape=in_shape))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=num_actions))

    nadam_opt = tf.keras.optimizers.Nadam(lr=learn_rate)
    model.compile(optimizer=nadam_opt, loss=tf.keras.losses.mean_squared_error)

    return model

"""Implementation of a Double Deep Q Network.

Attributes:
    batch_size: Number of random samples in experience buffer to learn from.
    discount: Discount factor to use in bellman update.
    exp_rate: Current rate at which agent will explore (random actions).
    min_exp_rate: Minimum rate at which agent will explore.
    decay_steps: Number of episodes before reaching min_exp_rate.
    decay_delta: Amount to subtract from exp_rate after each episode to reach
                 min_exp_rate.
    state_shape: Shape of input to the network.
    num_actions: Number of actions available to agent.
    learn_rate: Learning rate for bellman update.
    sync_rate: Number of episodes per sync of target weights and learning weights
    mem: A deque used to store past observations/experiences of agent and sample
         from for learning.
    max_mem: Maximum number of experiences to store.
    q: Main q network used for learning and predictions.
    target: Another q network used for updating q in bellman updates. This is
            the 'double' in ddqn, which helps eliminate variances that compound
            over time from bad early predictions.
"""
class DDQN(object):
    def __init__(self, in_shape, num_actions, learn_rate):
        """Initialize ddqn network including both networks and experience memory.

        Args:
            in_shape: Shape of input to network.
            num_actions: Number of actions agent is allowed to perform.
            learn_rate: Learning rate for optimization of network.
        """
        self.batch_size = 32
        self.discount = 0.99
        self.exp_rate = 1.0
        self.min_exp_rate = 0.001
        self.decay_steps = 20000
        self.decay_delta = (self.exp_rate - self.min_exp_rate) / self.decay_steps
        self.state_shape = in_shape
        self.num_actions = num_actions
        self.learn_rate = learn_rate
        self.sync_rate = 50

        self.mem = deque()
        self.max_mem = 50000

        self.q = dqn(self.state_shape, self.num_actions, self.learn_rate)
        self.target = dqn(self.state_shape, self.num_actions, self.learn_rate)

    def save_ddqn(self, f_name):
        """Save ddqn model.

        Only save q, since target will be synced when model is next loaded.

        Args:
            f_name: Name of file to save model to.
        """
        self.q.save_weights(f_name + '.h5')
        # self.q.save(f_name + '.h5')

    def load_ddqn(self, f_name):
        """Load ddqn model.

        Use same saved model for both networks.

        Args:
            f_name: Name of file model is loaded from.
        """
        self.q.load_weights(f_name + '.h5')
        self.target.load_weights(f_name + '.h5')
        # del self.q
        # del self.target
        # self.q = tf.keras.models.load_model(f_name + '.h5')
        # self.target = tf.keras.models.load_model(f_name + '.h5')

    def _decay_exp_rate(self):
        """Decay the exploration rate."""
        self.exp_rate -= self.decay_delta

    def _add_memory(self, states):
        """Add an ovservation to experience memory."""
        self.mem.append(states)

    def observe(self, states):
        """Add the experience to memory.

        Also decay exploration rate and remove oldest experience if memory full.

        Args:
            states: The states to add to memory (s1, a, r, s2, is_term).
        """
        self._add_memory(states)
        is_finished = states[-1]
        # If end of episode, decay the exploration rate
        if is_finished and self.exp_rate > self.min_exp_rate:
            self._decay_exp_rate()

        if len(self.mem) > self.max_mem:
            self.mem.popleft()

    def choose_action(self, s, rand=True):
        """Have network choose best action.

        Args:
            s: State to use for action prediction.
            rand: Whether or not to incorporate exploration chance.

        Returns:
            Agents best predicted action given the current state.
        """
        randnum = np.random.rand()
        if rand:
            if randnum > self.exp_rate:
                a = np.argmax(self.q.predict(s))
            else:
                a = random.randrange(self.num_actions)
        else:
            a = np.argmax(self.q.predict(s))

        return a

    def train_mem(self):
        """Perform a round of training for the agent.

        Samples experiences from memory and uses those to train from rather
        than sequential states. This helps the agent generalize its predictions,
        rather than training them based on a specific sequence of states.

        Learns using bellman updates to q-value functions/matrices.
        Specifically, the main q network is updated using the current states reward,
        and the discounted future reward prediction from the extra target network.

        Returns:
            Loss calculated from fitting the model.
        """
        # Sample batch_size experiences from memory
        sample_size = min(self.batch_size, len(self.mem))
        samples = random.sample(self.mem, sample_size)

        s1 = np.zeros(((sample_size,) + self.state_shape))
        a = []
        r = []
        s2 = np.zeros(((sample_size,) + self.state_shape))
        terminal = []

        # Separate the features from samples into individual vectors.
        for i in range(sample_size):
            s1[i, :, :, :] = samples[i][0]
            a.append(samples[i][1])
            r.append(samples[i][2])
            s2[i, :, :, :] = samples[i][3]
            terminal.append(samples[i][4])

        # Main q network expected future reward prediction from s1
        q_s1_t = self.q.predict(s1, batch_size=self.batch_size)
        # Main q network expected future reward prediction from s2
        q_s2_t = self.q.predict(s2, batch_size=self.batch_size)
        # Target q network expected future reward prediction from s2
        target_s2_t = self.target.predict(s2, batch_size=self.batch_size)

        for i in range(sample_size):
            # If end of episode, expected future reward from s1 is just reward observed
            if terminal[i]:
                q_s1_t[i, a[i]] = r[i]
            else:
                # Choose best action from Main s2 prediction (index with highest reward)
                action = np.argmax(q_s2_t[i])
                # Bellman update of Q(s1, a) using future discounted reward from
                # TQ(s2, action)
                q_s1_t[i, a[i]] = r[i] + self.discount * target_s2_t[i, action]

        # Optimize error between state s1 and Q(s1, *)
        loss = self.q.fit(s1, q_s1_t, batch_size=self.batch_size, verbose=0)
        return loss.history['loss']

    def sync_target_q(self):
        """Synchronize weights between main and target network."""
        print("Updating target...")
        self.target.set_weights(self.q.get_weights())

if __name__ == "__main__":
    # Set up tensorflow session and keras backend
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # Settings for learning all here
    begin_epochs = 0
    train_epochs = 400
    eps_per_epoch = 50
    target_update_freq = 10
    save_rate = 1
    learn_rate = 0.00025
    frame_w, frame_h = 64, 64
    n_frames = 4
    in_shape = (frame_w, frame_h, n_frames)
    visible_eps = 10
    load_model = True
    # load_model = False
    # cont_train = True
    cont_train = False

    # Initialize game and load given scenario
    game = DoomGame()
    game.load_config('./defend_the_center.cfg')
    # game.load_config('./deadly_corridor.cfg')
    game.set_window_visible(False)
    num_actions = game.get_available_buttons_size()
    actions = get_dummies(np.arange(num_actions)).values
    game.init()

    # Initialize network
    ddqn = DDQN(in_shape, num_actions, learn_rate)

    if load_model:
        # Load previous saved model and progress
        ddqn.load_ddqn("doom_ddqn")
        with open("exp.txt", "r") as f:
            exp = float(f.readline())
            epochs = int(f.readline())
            print("exp_rate: {}, epoch: {}".format(exp, epochs + 1))
            ddqn.exp_rate = exp
            begin_epochs = epochs

    for i in range(begin_epochs, train_epochs):
        # Don't train at all if model loaded and not continue training
        if load_model and not cont_train:
            break

        # Sync target and main network if current epoch multiple of frequency
        if i % target_update_freq == 0:
            ddqn.sync_target_q()

        # Save network if current epoch multiple of save rate
        if i % save_rate == 0:
            print("Saving...")
            ddqn.save_ddqn("doom_ddqn")

        print("Epoch {}".format(i + 1))
        print("-" * 50)
        losses = []
        # Get current exploration rate in case training interrupted to save it
        curr_exp_rate = ddqn.exp_rate
        # state = game.get_state()
        # # Get frame, process it, states are stacks of 4 temporal frames
        # frame = process_frame(state.screen_buffer, (frame_w, frame_h))
        # s1 = np.stack([frame, frame, frame, frame], axis=2)
        # s1 = s1[np.newaxis]
        try:
            for j in trange(eps_per_epoch, leave=False):
                game.new_episode()
                state = game.get_state()
                # Get frame, process it, states are stacks of 4 temporal frames
                frame = process_frame(state.screen_buffer, (frame_w, frame_h))
                s1 = np.stack([frame, frame, frame, frame], axis=2)
                s1 = s1[np.newaxis]
                while not game.is_episode_finished():
                    # Advance episode using networks action prediction
                    a = ddqn.choose_action(s1)
                    game.set_action(actions[a].tolist())
                    game.advance_action(n_frames)
                    state = game.get_state()
                    r = game.get_last_reward()

                    # If episode finished, s2 is None, otherwise append to end of stack
                    if game.is_episode_finished():
                        s2 = None
                    else:
                        frame = process_frame(state.screen_buffer, (frame_w, frame_h))
                        frame = frame[np.newaxis, :, :, np.newaxis]
                        s2 = np.append(frame, s1[:, :, :, :3], axis=3)

                    # Add observations/states to memory
                    ddqn.observe((s1, a, r, s2, game.is_episode_finished()))
                    loss = ddqn.train_mem()
                    losses.append(loss)

                    # Advance s1 to s2
                    s1 = s2
        finally:
            # If training interrupted or ended save experience rate and epoch
            with open("exp.txt", "w") as f:
                f.write("{}\n".format(curr_exp_rate))
                f.write("{}".format(i))

        print("Mean loss: {}".format(np.mean(losses)))

    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    # Restart game and see how agent performs visually
    # Will save final episode for replay
    for i in range(visible_eps):
        game.new_episode('episode.lmp')

        while not game.is_episode_finished():
            frame = process_frame(game.get_state().screen_buffer, (frame_w, frame_h))
            frame = np.stack([frame, frame, frame, frame], axis=2)
            frame = frame[np.newaxis]
            a = ddqn.choose_action(frame, rand=False)
            game.set_action(actions[a].tolist())
            for _ in range(n_frames):
                game.advance_action()

        score = game.get_total_reward()
        print("Total reward: {}".format(score))

    game.close()
