from serpent.game_api import GameAPI

from serpent.input_controller import KeyboardKey, KeyboardEvent, KeyboardEvents
from serpent.input_controller import MouseButton, MouseEvent, MouseEvents

from serpent.frame_grabber import FrameGrabber

import pytesseract

import serpent.cv

import numpy as np

import skimage.io
import skimage.util
import skimage.morphology
import skimage.segmentation
import skimage.measure

import math
import time
import random


from PIL import Image



class CODAPI(GameAPI):
    def __init__(self, game=None):
        super().__init__(game=game)

        self.game_inputs = {
            "MOVEMENT": {
                "WALK LEFT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
                ],
                "STRAFE LEFT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A),
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_SHIFT)
                ],
                "SPRINT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_SHIFT)
                ],
                "WALK": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W)
                ],
                "STRAFE RIGHT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D),
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_SHIFT)
                ],
                "WALK RIGHT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
                ],
                "BACK": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S)
                ],
                "JUMP": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_SPACE)
                ],
                "CROUCH": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_C)
                ],
                "PHRONE": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_CTRL)
                ],
                "STOPPED": []
            },
            "COMBAT": {
                "RELOAD": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_R)
                ],
                "ABILTY": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_B)
                ],
                "GRENADE": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_G)
                ],
                "MELEE": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_E)
                ],
                "USE": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_F)
                ],
                "NON LETHAL": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_Q)
                ],
                "SECONDARY": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_1)
                ],
                "Neutral": []
            },
            "CURSOR": {
                "MOVE MOUSE X": [
                ],
                "MOVE MOUSE Y": [
                ],
                "MOVE MOUSE XY": [
                ],
                "MOVE MOUSE X2": [
                ],
                "MOVE MOUSE Y2": [
                ],
                "MOVE MOUSE XY2": [
                ],
                "MOVE MOUSE XY3": [
                ],
                "MOVE MOUSE XY4": [
                ],
                "IDLE_MOUSE": []
            }
        }
    def num_there(self, s):
        return any(i.isdigit() for i in s)
    def parse_ammo(self, image):
        crop_area = (951, 1645, 998, 1756)
        img = Image.frombytes("RGB", image.size, image.bgra, "raw", "BGRX")
        cropped_img = img.crop(crop_area)
        ocr_result = pytesseract.image_to_string(cropped_img, lang='eng', \
           config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        if self.num_there(ocr_result):
            ocr_result_int = int(ocr_result)
            return ocr_result_int <= 8
        else:
            return False
        
    def get_health(self, image):
        img = Image.frombytes('RGB', image.size, image.rgb)
        red_O = 0
        for red in img.getdata():
            if red == (117,54,34):
                red_O += 1
        return red_O >= 1
