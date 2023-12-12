import numpy as np
import cv2 as cv
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import cam


class HandGestureDetector:
    # initialize camera and drawing points
    def __init__(self, mp_ar) -> None:
        self.init_options()
        self.mp_ar = mp_ar
        self.cp = cam.CameraProcessor(mp_ar)
        self.fg_pts = deque(maxlen=64)

    # initialize default options of the hand gesture detection of landmark
    def init_options(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.hand_landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=5,
            circle_radius=5,
        )
        self.hand_connection_drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=10,
            circle_radius=10,
        )

        base_options = python.BaseOptions(
            model_asset_path="models/gesture_recognizer.task"
        )
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    # show modified frame
    def set_output(self):
        cv.imshow("Output", self.cp.rescale_frame(percent=130))

    # detects the hand gesture using landmark recognition
    def detect(self, results_hand, frames_to_finish, idx_to_coordinates):
        take_screenshot = False
        # create a media pipe image from the current frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.cp.get_frame())

        # recognize the hand gestures using landmark
        recognition_result = self.recognizer.recognize(mp_image)

        if len(recognition_result.gestures) != 0:
            # choose the best detected gesture
            top_gesture = recognition_result.gestures[0][0]

            # add detected gesture text and detection score to the frame
            gesture_prediction = (
                f"{top_gesture.category_name} ({top_gesture.score:.2f})"
            )
            self.cp.text_gesture(gesture_prediction)

            if results_hand.multi_hand_landmarks:
                # show detection joints of multiple hands detected
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=self.cp.get_frame(),
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.hand_landmark_drawing_spec,
                        connection_drawing_spec=self.hand_connection_drawing_spec,
                    )

            # process commands according to the detected gesture
            if top_gesture.category_name == "Pointing_Up":
                idx_to_coordinates = self.get_idx_to_coordinates(results_hand)

                # append the index finger tip coordinates to the drawing points
                if 8 in idx_to_coordinates:
                    self.fg_pts.appendleft(idx_to_coordinates[8])  # Index Finger

            elif top_gesture.category_name == "Victory":
                # set the remaining frames to be processed until program closure
                if frames_to_finish is None:
                    frames_to_finish = 200
                self.set_output()

            elif top_gesture.category_name == "Thumb_Up":
                # setting this variable as true helps the caller function to understand
                # this frame should be chosen for screenshot
                take_screenshot = True

        return take_screenshot, frames_to_finish, idx_to_coordinates

    # fills the coordinates of the frame with color
    def draw_finger_points(self):
        for i in range(1, len(self.fg_pts)):
            if self.fg_pts[i - 1] is None or self.fg_pts[i] is None:
                continue
            thickness = int(np.sqrt(len(self.fg_pts) / float(i + 1)) * 4.5)
            cv.line(
                self.cp.get_frame(),
                self.fg_pts[i - 1],
                self.fg_pts[i],
                (0, 255, 0),
                thickness,
            )

    # returns indices of the points mapped with landmark detected pixels
    def get_idx_to_coordinates(
        self, results, VISIBILITY_THRESHOLD=0.5, PRESENCE_THRESHOLD=0.5
    ):
        idx_to_coordinates = {}
        image_rows, image_cols, _ = self.cp.get_frame().shape
        try:
            for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                if (
                    landmark.HasField("visibility")
                    and landmark.visibility < VISIBILITY_THRESHOLD
                ) or (
                    landmark.HasField("presence")
                    and landmark.presence < PRESENCE_THRESHOLD
                ):
                    continue
                landmark_px = _normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image_cols, image_rows
                )
                if landmark_px:
                    idx_to_coordinates[idx] = landmark_px
        except:
            pass
        return idx_to_coordinates

    def run(self):
        masterpiece = None
        frames_to_finish = None

        self.cp.start()

        while self.cp.is_capturing():
            idx_to_coordinates = {}
            self.cp.frame_read()
            self.cp.frame_flip()

            take_screenshot = False
            if frames_to_finish is not None:
                # program should be closed after certain frame processing
                if frames_to_finish == 0:
                    break
                # gray the frame and set the goodbye text
                self.cp.to_gray()
                self.cp.text_goodbye()

                frames_to_finish -= 1
            else:
                # hand processor needs rgb color
                self.cp.to_rgb()
                results_hand = self.hands.process(self.cp.get_frame())

                self.cp.make_frame_writable()

                # change back to bgr to be used fro detector
                self.cp.to_bgr()

                take_screenshot, frames_to_finish, idx_to_coordinates = self.detect(
                    results_hand, frames_to_finish, idx_to_coordinates
                )

            self.draw_finger_points()

            if take_screenshot == True:
                masterpiece = self.cp.get_frame()
                self.cp.text_screenshot()

            if frames_to_finish is None and masterpiece is not None:
                self.cp.set_masterpiece(masterpiece)

            self.set_output()
            if cv.waitKey(5) & 0xFF == 27:
                break
        self.hands.close()
        self.cp.stop()
