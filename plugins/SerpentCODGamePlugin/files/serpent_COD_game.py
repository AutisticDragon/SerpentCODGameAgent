from serpent.game import Game

from .api.api import CODAPI

from serpent.utilities import Singleton

from .environments.cod_environment import StartRegionsEnvironment
from .environments.common import StartRegions

import time

from serpent.input_controller import InputControllers

class SerpentCODGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["input_controller"] = InputControllers.NATIVE_WIN32

        kwargs["window_name"] = "Call of Duty®: Modern Warfare®"

        
        
        kwargs["executable_path"] = "F:/Battle Net/Call of Duty Modern Warfare/ModernWarfare.exe"
        
        

        super().__init__(**kwargs)

        self.api_class = CODAPI
        self.api_instance = None

        self.environments = {
            "GAME": StartRegionsEnvironment
        }

        self.environment_data = {
            "START_REGIONS": StartRegions
        }

        self.frame_transformation_pipeline_string = "RESIZE:100x100|GRAYSCALE|FLOAT"

    @property
    def screen_regions(self):
        regions = {
            "AMMO": (951, 1645, 998, 1756),
            "CUSTOM_GAME": (37, 61, 95, 474),
            "XP": (471, 1023, 495, 1096)
        }

        return regions
    def after_launch(self):
        self.is_launched = True

        current_attempt = 1

        while current_attempt <= 100:
            self.window_id = self.window_controller.locate_window(self.window_name)

            if self.window_id not in [0, "0"]:
                break

            time.sleep(0.1)

        time.sleep(0.5)

        if self.window_id in [0, "0"]:
            raise SerpentError("Game window not found...")

        self.window_controller.move_window(self.window_id, 0, 0)

        self.dashboard_window_id = self.window_controller.locate_window("Serpent.AI Dashboard")

        # TODO: Test on macOS and Linux
        if self.dashboard_window_id is not None and self.dashboard_window_id not in [0, "0"]:
            self.window_controller.bring_window_to_top(self.dashboard_window_id)

        self.window_controller.focus_window(self.window_id)

        self.window_geometry = self.extract_window_geometry()

        print(self.window_geometry)