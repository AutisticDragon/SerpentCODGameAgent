from serpent.game_agent import GameAgent

from serpent.enums import InputControlTypes

from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey

from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent
import pytesseract

import numpy as np

from serpent.config import config

from serpent.logger import Loggers

import serpent.cv
import signal
import time
import random

from mss import mss

from PIL import Image
import os

import cv2
import skimage

import pyautogui
import pywinauto
import keyboard

Wd, Hd = 1920, 1080
ACTIVATION_RANGE = 300
YOLO_DIRECTORY = "models"
CONFIDENCE = 0.36
THRESHOLD = 0.22
ACTIVATION_RANGE = 350
labelsPath = os.path.sep.join([YOLO_DIRECTORY, "coco-dataset.labels"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.weights"])
configPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
origbox = (int(Wd/2 - ACTIVATION_RANGE/2),
    int(Hd/2 - ACTIVATION_RANGE/2),
    int(Wd/2 + ACTIVATION_RANGE/2),
    int(Hd/2 + ACTIVATION_RANGE/2))
def set_pos(x, y):
    current_x, current_y = pyautogui.position()
    new_x = current_x + x
    new_y = current_y + y
    pywinauto.mouse.move(coords=(new_x, new_y))
def set_pos_aimbot(x, y):
    pywinauto.mouse.move(coords=(x, y))
class SerpentCODGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_play_pause

    def setup_play(self):
        self.environment = self.game.environments["GAME"](
            game_api=self.game.api,
            input_controller=self.input_controller,
            episodes_per_startregions_track=100000000000
        )

        self.game_inputs = [
            {
                "name": "CONTROLS",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["MOVEMENT", "COMBAT", "CURSOR"])
            }
        ]
        self.agent = RainbowDQNAgent(
            "COD",
            game_inputs=self.game_inputs,
            callbacks=dict(
                after_observe=self.after_agent_observe,
                before_update=self.before_agent_update,
                after_update=self.after_agent_update
                ),
            rainbow_kwargs=dict(
                replay_memory_capacity=250000,
                observe_steps=100,
                batch_size=10,
                save_steps=300,
                model="datasets/rainbow_dqn_COD.pth"
                ),
            logger=Loggers.COMET_ML,
            logger_kwargs=dict(
                api_key="api_key_from_comet_ml",
                project_name="serpent-ai-cod",
                reward_func=self.reward
            )
            )
        self.analytics_client.track(event_key="COD", data={"name": "COD"})
        self.agent.logger.experiment.log_other("game", "COD")
        self.environment.new_episode(maximum_steps=350)  # 5 minutes
        self.overs = 0
        self.input_non_lethal = False
    def handle_play(self, game_frame, game_frame_pipeline):
        self.paused_at = None
        with mss() as sct:
            monitor_var = sct.monitors[1]
            monitor = sct.grab(monitor_var)
            valid_game_state = self.environment.update_startregions_state(monitor)
        if not valid_game_state:
            return None

        reward, over_boolean = self.reward(self.environment.startregions_state, 1.0)
        terminal = over_boolean

        self.agent.observe(reward=reward, terminal=terminal)

        if not terminal:
            game_frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
            agent_actions = self.agent.generate_actions(game_frame_buffer)
            print("Current Action: ")
            print(agent_actions)
            str_agent_actions = str(agent_actions)
            if "MOVE MOUSE X" in str_agent_actions:
                set_pos(200, 0)
            if "MOVE MOUSE Y" in str_agent_actions:
                set_pos(0, 200)
            if "MOVE MOUSE XY" in str_agent_actions:
                set_pos(100, 100)
            if "MOVE MOUSE X2" in str_agent_actions:
                set_pos(-200, 0)
            if "MOVE MOUSE Y2" in str_agent_actions:
                set_pos(0, -200)
            if "MOVE MOUSE XY2" in str_agent_actions:
                set_pos(-100, -100)
            if "MOVE MOUSE XY3" in str_agent_actions:
                set_pos(-100, 100)
            if "MOVE MOUSE XY4" in str_agent_actions:
                set_pos(100, -100)
            if "LETHAL" in str_agent_actions:
                self.input_non_lethal = True
            self.human()
            self.human()
            self.human()
            self.human()
            self.environment.perform_input(agent_actions)
        else:
            self.environment.clear_input()
            self.agent.reset()

            time.sleep(30)
            #To Do
            #Choose Loadout (Meduim Range)
            self.environment.end_episode()
            self.environment.new_episode(maximum_steps=350)
            print("New Episode")
    def handle_play_pause(self):
        self.input_controller.handle_keys([])
    def num_there(self, s):
        return any(i.isdigit() for i in s)

    def get_health(self, image):
        img = Image.frombytes('RGB', image.size, image.rgb)
        red_O = 0
        for red in img.getdata():
            if red == (117,54,34):
                red_O += 1
        return red_O
    def get_xp(self, image_xp):
        img = Image.frombytes('RGB', image_xp.size, image_xp.rgb)
        pixels = 0
        for pixel in img.getdata():
            if pixel == (255,194,21):
                pixels += 1
        return pixels
    def is_startregions_over(self, image):
        image = Image.frombytes("RGB", image.size, image.bgra, "raw", "BGRX")
        ocr_result = pytesseract.image_to_string(image, lang='eng')
        print("Text: ")
        print(ocr_result)
        if "KILLCAM" in ocr_result:
            return True
        else:
            return False
    def human(self):
        with mss() as sct:
            W, H = None, None
            frame = np.array(sct.grab(origbox))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


            if W is None or H is None:
                (H, W) = frame.shape[: 2]

            frame = cv2.UMat(frame)
            blob = cv2.dnn.blobFromImage(frame, 1 / 260, (150, 150),
                swapRB=False, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = 0
                    confidence = scores[classID]
                    if confidence > CONFIDENCE:
                        box = detection[0: 4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
            if len(idxs) > 0:
                bestMatch = confidences[np.argmax(confidences)]

                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw target dot on the frame
                    cv2.circle(frame, (int(x + w / 2), int(y + h / 5)), 5, (0, 0, 255), -1)

                    # draw a bounding box rectangle and label on the frame
                    # color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y),
                                    (x + w, y + h), (0, 0, 255), 2)

                    text = "TARGET {}%".format(int(confidences[i] * 100))
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if bestMatch == confidences[i]:
                        mouseX = origbox[0] + (x + w/1.5)
                        mouseY = origbox[1] + (y + h/5)
                        mouseX = int(round(mouseX))
                        mouseY = int(round(mouseY))
                        set_pos_aimbot(mouseX, mouseY)
                        pywinauto.mouse.click(button='left', coords=(mouseX, mouseY))
                        pywinauto.mouse.click(button='left', coords=(mouseX, mouseY))
                        pywinauto.mouse.click(button='left', coords=(mouseX, mouseY))
                        pywinauto.mouse.click(button='left', coords=(mouseX, mouseY))
                        pywinauto.mouse.click(button='left', coords=(mouseX, mouseY))
                        pywinauto.mouse.click(button='left', coords=(mouseX, mouseY))
    def reward(self, game_state, object_reward_func):
        with mss() as sct:
            image = sct.grab(sct.monitors[1])
            value = self.get_health(image)
            print("Health: ")
            print(value * -1)
            monitor = {"top": 452, "left": 1000, "width": 144, "height": 51, "mon": 1}
            image_xp = sct.grab(monitor)
            xp = self.get_xp(image_xp)
            monitor_custom_game = {"top": 47, "left": 50, "width": 230, "height": 66, "mon": 1}
            image_over = sct.grab(monitor_custom_game)
            over_check = self.is_startregions_over(image_over)
            reward = 0.0
            over = False
            if over_check:
                reward = 0.0
                self.overs += 1
                if self.overs >= 6:
                    print("Game Over")
                    over = True
                    self.overs = 0
                else:
                    over = False
            else:
                reward = 0.0
                if value >= 1:
                    reward += -1.5
                elif value < 1:
                    reward += 0.0
                if value >= 1 and self.input_non_lethal:
                    reward += 2.5
                    self.input_non_lethal = False
                if xp >= 7:
                    reward += 3.0
            print("Reward: ")
            print(reward)
            return reward, over
    def after_agent_observe(self):
        self.environment.episode_step()

    def before_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(1)

    def after_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(3)
