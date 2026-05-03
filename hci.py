import numpy as np
import cv2 as cv
import os
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cam


def normalized_to_pixel_coordinates(
    normalized_x, normalized_y, image_width, image_height
):
    """Convert normalized landmark coordinates to bounded pixel coordinates."""
    if normalized_x is None or normalized_y is None:
        return None
    x_px = min(max(int(normalized_x * image_width), 0), image_width - 1)
    y_px = min(max(int(normalized_y * image_height), 0), image_height - 1)
    return x_px, y_px


class HandGestureDetector:
    # initialize camera and drawing points
    def __init__(self, mp_ar) -> None:
        self.init_options()
        self.mp_ar = mp_ar
        self.cp = cam.CameraProcessor(mp_ar)
        self.brush_palette = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
        }
        self.brush_name = "green"
        self.brush_color = self.brush_palette[self.brush_name]
        self.brush_thickness = 5
        self.eraser_mode = False
        self.eraser_radius = 28
        self.status_message = (
            "U undo | R redo | C clear | S save | E eraser | 1-4 colors | [ ] size"
        )
        self.status_message_ticks = 0
        self.strokes = []
        self.current_stroke = self._new_stroke()
        self.redo_strokes = []

    # initialize default options of the hand gesture detection of landmark
    def init_options(self):
        base_options = python.BaseOptions(
            model_asset_path="models/gesture_recognizer.task"
        )
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def clear_drawing(self):
        """Remove all stored strokes and reset redo history."""
        self.current_stroke = self._new_stroke()
        self.strokes.clear()
        self.redo_strokes.clear()

    def _new_stroke(self):
        return {
            "points": [],
            "color": self.brush_color,
            "thickness": self.brush_thickness,
        }

    def set_brush_color(self, name):
        """Select a preset brush color for newly started strokes."""
        if name not in self.brush_palette:
            return
        self.brush_name = name
        self.brush_color = self.brush_palette[name]
        self.eraser_mode = False
        self.set_status_message(f"Brush: {name}")

    def set_brush_thickness(self, thickness):
        """Clamp and apply a new brush thickness for newly started strokes."""
        self.brush_thickness = max(1, min(25, thickness))
        self.set_status_message(f"Thickness: {self.brush_thickness}")

    def increase_thickness(self):
        self.set_brush_thickness(self.brush_thickness + 1)

    def decrease_thickness(self):
        self.set_brush_thickness(self.brush_thickness - 1)

    def toggle_eraser_mode(self):
        """Switch between brush mode and eraser mode."""
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.set_status_message("Eraser mode on")
        else:
            self.set_status_message(f"Brush: {self.brush_name}")

    def finalize_current_stroke(self):
        """Commit the active point sequence as a completed stroke."""
        if self.current_stroke["points"]:
            self.strokes.append(
                {
                    "points": self.current_stroke["points"].copy(),
                    "color": self.current_stroke["color"],
                    "thickness": self.current_stroke["thickness"],
                }
            )
            self.current_stroke = self._new_stroke()

    def add_point_to_stroke(self, point):
        """Append a new drawing point and invalidate redo history when needed."""
        if point is None:
            return
        if self.eraser_mode:
            self.erase_at_point(point)
            return
        if not self.current_stroke["points"]:
            self.redo_strokes.clear()
            self.current_stroke["color"] = self.brush_color
            self.current_stroke["thickness"] = self.brush_thickness
        self.current_stroke["points"].append(point)

    def erase_at_point(self, point):
        """Remove points close to the eraser cursor from all stored strokes."""
        if point is None:
            return
        self.redo_strokes.clear()

        def keep_points(stroke):
            filtered_points = []
            for stroke_point in stroke["points"]:
                distance = np.linalg.norm(np.array(stroke_point) - np.array(point))
                if distance > self.eraser_radius:
                    filtered_points.append(stroke_point)
            stroke["points"] = filtered_points

        keep_points(self.current_stroke)
        for stroke in self.strokes:
            keep_points(stroke)
        self.strokes = [stroke for stroke in self.strokes if stroke["points"]]

    def undo_last_stroke(self):
        """Undo the current active stroke or move the last completed stroke to redo."""
        if self.current_stroke["points"]:
            self.current_stroke = self._new_stroke()
            return
        if self.strokes:
            self.redo_strokes.append(self.strokes.pop())

    def redo_last_stroke(self):
        """Restore the most recently undone stroke."""
        if self.redo_strokes:
            self.strokes.append(self.redo_strokes.pop())

    def save_composition(self):
        """Export the current composited frame to the exports directory."""
        os.makedirs("exports", exist_ok=True)
        filename = os.path.join(
            "exports", datetime.now().strftime("webcam_drawing_%Y%m%d_%H%M%S.png")
        )
        cv.imwrite(filename, self.cp.get_frame().copy())
        self.set_status_message(f"Saved {os.path.basename(filename)}", ticks=45)
        return filename

    def set_status_message(self, message, ticks=30):
        """Display a short-lived status banner on top of the frame."""
        self.status_message = message
        self.status_message_ticks = ticks

    def close_program(self, frames_to_finish, countdown=200):
        if frames_to_finish is None:
            frames_to_finish = countdown
        return frames_to_finish

    def handle_gesture(self, gesture_name, frames_to_finish):
        """Map gesture labels to application actions."""
        take_screenshot = False
        clear_drawing = False
        undo_stroke = False

        if gesture_name == "Victory":
            frames_to_finish = self.close_program(frames_to_finish)
            self.set_output()

        elif gesture_name == "Thumb_Up":
            # setting this variable as true helps the caller function to understand
            # this frame should be chosen for screenshot
            take_screenshot = True

        elif gesture_name == "Open_Palm":
            clear_drawing = True

        elif gesture_name == "Thumb_Down":
            undo_stroke = True

        return take_screenshot, frames_to_finish, clear_drawing, undo_stroke

    def draw_hand_landmarks(self, hand_landmarks):
        """Render hand landmarks and connections from the gesture result."""
        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        points = []
        image_rows, image_cols, _ = self.cp.get_frame().shape

        for landmark in hand_landmarks:
            landmark_px = normalized_to_pixel_coordinates(
                landmark.x, landmark.y, image_cols, image_rows
            )
            points.append(landmark_px)
            if landmark_px is not None:
                cv.circle(self.cp.get_frame(), landmark_px, 4, (0, 255, 255), -1)

        for connection in connections:
            start_idx = connection.start
            end_idx = connection.end
            if start_idx >= len(points) or end_idx >= len(points):
                continue
            start_point = points[start_idx]
            end_point = points[end_idx]
            if start_point is None or end_point is None:
                continue
            cv.line(self.cp.get_frame(), start_point, end_point, (0, 255, 0), 2)

    # show modified frame
    def set_output(self):
        """Show the current output frame in a rescaled window."""
        cv.imshow("Output", self.cp.rescale_frame(percent=130))

    # detects the hand gesture using landmark recognition
    def detect(self, recognition_result, frames_to_finish, idx_to_coordinates):
        """Run gesture-specific actions for the current recognition result."""
        take_screenshot = False
        undo_stroke = False

        if len(recognition_result.gestures) != 0:
            # choose the best detected gesture
            top_gesture = recognition_result.gestures[0][0]

            # add detected gesture text and detection score to the frame
            gesture_prediction = (
                f"{top_gesture.category_name} ({top_gesture.score:.2f})"
            )
            self.cp.text_gesture(gesture_prediction)

            if recognition_result.hand_landmarks:
                # show detection joints of multiple hands detected
                for hand_landmarks in recognition_result.hand_landmarks:
                    self.draw_hand_landmarks(hand_landmarks)

            # process commands according to the detected gesture
            if top_gesture.category_name == "Pointing_Up":
                idx_to_coordinates = self.get_idx_to_coordinates(recognition_result)

                # append the index finger tip coordinates to the drawing points
                if 8 in idx_to_coordinates:
                    self.add_point_to_stroke(idx_to_coordinates[8])  # Index Finger
            else:
                self.finalize_current_stroke()
                take_screenshot, frames_to_finish, clear_drawing, undo_stroke = (
                    self.handle_gesture(top_gesture.category_name, frames_to_finish)
                )

                if clear_drawing:
                    self.clear_drawing()
                    self.set_status_message("Canvas cleared")

                if undo_stroke:
                    self.undo_last_stroke()
                    self.set_status_message("Undo last stroke")

        return take_screenshot, frames_to_finish, idx_to_coordinates, undo_stroke

    # fills the coordinates of the frame with color
    def draw_stroke(self, stroke):
        """Draw one completed or in-progress stroke on the current frame."""
        if not stroke["points"]:
            return
        color = stroke["color"]
        thickness = stroke["thickness"]
        points = stroke["points"]
        if len(points) == 1:
            cv.circle(self.cp.get_frame(), points[0], max(1, thickness // 2), color, -1)
            return
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv.line(
                self.cp.get_frame(),
                points[i - 1],
                points[i],
                color,
                thickness,
            )

    def draw_finger_points(self):
        """Draw all completed strokes plus the active in-progress stroke."""
        for stroke in self.strokes:
            self.draw_stroke(stroke)
        self.draw_stroke(self.current_stroke)

    def draw_controls(self):
        """Overlay the available keyboard controls and status banner."""
        cv.putText(
            self.cp.get_frame(),
            "U undo  R redo  C clear  S save  E eraser  1-4 colors  [ ] size  Esc exit",
            (10, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        mode_text = (
            f"Mode: {'eraser' if self.eraser_mode else 'brush'} | "
            f"Color: {self.brush_name} | Size: {self.brush_thickness}"
        )
        cv.putText(
            self.cp.get_frame(),
            mode_text,
            (10, 45),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        if self.status_message_ticks > 0:
            cv.putText(
                self.cp.get_frame(),
                self.status_message,
                (10, self.cp.get_frame().shape[0] - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            self.status_message_ticks -= 1

    # returns indices of the points mapped with landmark detected pixels
    def get_idx_to_coordinates(self, results):
        """Extract visible hand landmark coordinates from a gesture result."""
        idx_to_coordinates = {}
        image_rows, image_cols, _ = self.cp.get_frame().shape
        try:
            for idx, landmark in enumerate(results.hand_landmarks[0]):
                landmark_px = normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image_cols, image_rows
                )
                if landmark_px:
                    idx_to_coordinates[idx] = landmark_px
        except (AttributeError, IndexError):
            pass
        return idx_to_coordinates

    def run(self):
        """Main processing loop for capture, recognition, drawing, and export."""
        masterpiece = None
        frames_to_finish = None

        self.cp.start()

        while self.cp.is_capturing():
            idx_to_coordinates = {}
            undo_stroke = False
            if not self.cp.frame_read():
                break
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
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=self.cp.get_frame()
                )
                recognition_result = self.recognizer.recognize(mp_image)

                # change back to bgr to be used for annotation
                self.cp.to_bgr()

                (
                    take_screenshot,
                    frames_to_finish,
                    idx_to_coordinates,
                    undo_stroke,
                ) = self.detect(
                    recognition_result, frames_to_finish, idx_to_coordinates
                )

            if undo_stroke:
                self.undo_last_stroke()

            self.draw_finger_points()
            self.draw_controls()

            if take_screenshot == True:
                masterpiece = self.cp.get_frame().copy()
                self.cp.text_screenshot()

            if frames_to_finish is None and masterpiece is not None:
                self.cp.set_masterpiece(masterpiece)

            self.set_output()
            key = cv.waitKey(5) & 0xFF
            if key == ord("u"):
                self.undo_last_stroke()
                self.set_status_message("Undo last stroke")
            elif key == ord("r"):
                self.redo_last_stroke()
                self.set_status_message("Redo last stroke")
            elif key == ord("c"):
                self.clear_drawing()
                self.set_status_message("Canvas cleared")
            elif key == ord("s"):
                self.save_composition()
            elif key == ord("e"):
                self.toggle_eraser_mode()
            elif key == ord("1"):
                self.set_brush_color("red")
            elif key == ord("2"):
                self.set_brush_color("green")
            elif key == ord("3"):
                self.set_brush_color("blue")
            elif key == ord("4"):
                self.set_brush_color("yellow")
            elif key == ord("["):
                self.decrease_thickness()
            elif key == ord("]"):
                self.increase_thickness()
            elif key == 27:
                break
        self.cp.stop()
