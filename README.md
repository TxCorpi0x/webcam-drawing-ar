# Webcam Drawing AR

An augmented reality drawing system that combines webcam capture, hand gesture recognition, and ArUco marker-based compositing to let a user draw in the air and project the result back into the live camera stream.

## Abstract

This project implements a real-time human-computer interaction prototype in which hand gestures are used as the primary control channel. The application detects the user's hand, recognizes a small gesture vocabulary with MediaPipe, stores fingertip motion as a freehand trajectory, and overlays the captured drawing onto a master frame using ArUco markers and perspective transformation. The result is an interactive webcam experience suitable for academic demonstration in computer vision, mixed reality, and gesture-based interfaces.

## **Step 1. Problem Statement**

The system addresses the problem of creating a lightweight drawing interface without a physical stylus, mouse, or touchscreen. Instead of direct pointer input, it uses hand gestures detected from a standard webcam to trigger drawing, capture, and exit behaviors.

## **Step 2. Objectives**

The implementation is designed to:

- detect a hand from live webcam frames,
- recognize gesture classes with a machine-learning model,
- convert fingertip motion into a visible stroke path,
- capture a frame as the drawing canvas,
- reinsert the captured canvas into the live stream through marker-based augmentation,
- provide a simple visual interaction loop that can be demonstrated in class or in a report.

## **Step 3. Method Overview**

The application is organized around three cooperating modules:

- [main.py](main.py) initializes the augmented-reality pipeline and starts the gesture loop.
- [hci.py](hci.py) performs hand-landmark tracking, gesture recognition, stroke construction, and interaction logic.
- [cam.py](cam.py) handles webcam capture, frame conversion, and on-screen annotations.
- [ar.py](ar.py) performs ArUco marker detection and perspective warping so the captured drawing can be fitted into the scene.

The gesture recognizer uses the model stored in [models/gesture_recognizer.task](models/gesture_recognizer.task). The AR overlay depends on four visible ArUco markers being present in the camera view.

## **Step 4. Interaction Protocol**

The system currently interprets the following gestures:

- `Pointing_Up`: records the index-finger tip position and extends the drawing path.
- `Thumb_Up`: captures the current frame as the masterpiece snapshot.
- `Victory`: starts the closing sequence and displays a farewell message for a short countdown.
- `Open_Palm`: clears the current drawing canvas.
- `Thumb_Down`: undoes the last completed stroke.

Keyboard shortcuts are also available for a more mature editing workflow:

- `u`: undo the last stroke,
- `r`: redo the last undone stroke,
- `c`: clear the drawing canvas,
- `s`: save the current composition as a PNG file.
- `e`: toggle eraser mode,
- `1`: switch to red brush,
- `2`: switch to green brush,
- `3`: switch to blue brush,
- `4`: switch to yellow brush,
- `[`: decrease brush thickness,
- `]`: increase brush thickness.

The program also exits when the user presses `Esc`.

## **Step 5. Requirements**

You need:

- Python 3.9 or newer,
- a working webcam,
- four visible ArUco markers placed in the camera scene,
- `opencv-contrib-python` for `cv2.aruco`,
- `mediapipe` for hand tracking and gesture recognition,
- `numpy` for array and image operations.

## **Step 6. Installation**

1. Create a virtual environment in the project root:

```bash
python3 -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install the runtime dependencies from [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

4. Keep the repository layout unchanged so the gesture model remains reachable at [models/gesture_recognizer.task](models/gesture_recognizer.task).

## **Step 7. Execution**

Run the application from the repository root:

```bash
python main.py
```

The program opens the default webcam device, which is camera index `0` in [cam.py](cam.py). If you need another device, change the index in `CameraProcessor.start()`.

## **Step 8. System Behavior**

Once running, the pipeline follows this flow:

1. Capture a frame from the webcam.
2. Flip the image horizontally for mirror-like interaction.
3. Run hand-landmark detection in RGB space.
4. Recognize the gesture from the current hand image.
5. If the gesture is `Pointing_Up`, extend the active stroke history with the index-finger tip coordinates.
6. If the gesture is `Thumb_Up`, store the current frame as the drawing snapshot.
7. If the gesture is `Victory`, initiate shutdown and show a goodbye message.
8. If a snapshot exists and the AR markers are visible, warp the snapshot into the marker quadrilateral.
9. If the user triggers undo, redo, clear, or save, update the stroke history or export the current frame accordingly.
10. If the user changes brush settings or toggles eraser mode, apply the new drawing style to future strokes.

## **Step 9. Project Structure**

- [main.py](main.py): application entry point.
- [hci.py](hci.py): gesture recognition, landmark processing, and drawing loop.
- [cam.py](cam.py): webcam abstraction and frame utilities.
- [ar.py](ar.py): marker-based perspective fitting and overlay composition.
- [models/gesture_recognizer.task](models/gesture_recognizer.task): MediaPipe gesture model asset.
- [Report/HCI-AR.pdf](Report/HCI-AR.pdf): academic report associated with the project.

## **Step 10. Notes and Limitations**

- The AR overlay only appears when the required ArUco marker IDs are detected in the expected arrangement.
- The drawing path now keeps stroke history and per-stroke brush settings so the user can undo, redo, clear, export, and switch between colors or eraser mode.
- The program depends on real-time camera quality, lighting, and hand visibility.

## **Step 11. Citation and Academic Use**

If you are presenting or submitting this work, cite the implementation as a real-time gesture-based augmented reality drawing prototype and reference the PDF report in [Report/HCI-AR.pdf](Report/HCI-AR.pdf).
