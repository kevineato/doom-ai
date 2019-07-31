from time import sleep
from vizdoom import *

sleep_time = 1.0 / DEFAULT_TICRATE

game = DoomGame()
game.load_config('defend_the_center.cfg')
game.init()

game.replay_episode('episode.lmp')
sleep(1.0)

while not game.is_episode_finished():
    sleep(sleep_time)
    game.advance_action()

game.close()