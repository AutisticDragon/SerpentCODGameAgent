from serpent.environment import Environment

from serpent.input_controller import KeyboardKey

from serpent.utilities import SerpentError

import time
import collections

import numpy as np


class StartRegionsEnvironment(Environment):

    def __init__(self, game_api=None, input_controller=None, episodes_per_startregions_track=5):
        super().__init__("COD Environment", game_api=game_api, input_controller=input_controller)

        self.episodes_per_startregions_track = episodes_per_startregions_track

        self.reset()

    @property
    def new_episode_data(self):
        return {}

    @property
    def end_episode_data(self):
        return {}

    def new_episode(self, maximum_steps=None, reset=False):
        self.reset_startregions_state()

        time.sleep(1)

        if (self.episode + 1) % self.episodes_per_startregions_track == 0:
            self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        else:
            self.input_controller.tap_key(KeyboardKey.KEY_R)

        time.sleep(8)

        super().new_episode(maximum_steps=maximum_steps, reset=reset)

    def end_episode(self):
        super().end_episode()

    def reset(self):
        self.reset_startregions_state()
        super().reset()

    def reset_startregions_state(self):
        self.startregions_state = {
            "ammo_levels": False,
            "health_levels": False
        }

    def update_startregions_state(self, image):

        self.startregions_state["ammo_levels"] = self.game_api.parse_ammo(image)
        self.startregions_state["health_levels"] = self.game_api.get_health(image)

        return True