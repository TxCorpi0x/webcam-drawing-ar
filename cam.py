import cv2 as cv


class CameraProcessor:
    def __init__(self, mp_ar) -> None:
        self.mp_ar = mp_ar
        self.masterpiece_scale = 100

    def start(self):
        self.cap = cv.VideoCapture(0)

    def stop(self):
        self.cap.release()

    def is_capturing(self):
        return self.cap.isOpened()

    def frame_read(self):
        ret, self.frame = self.cap.read()

    def get_frame(self):
        return self.frame

    def set_frame(self, frame):
        self.frame = frame

    def set_masterpiece(self, masterpiece):
        ar_image = self.mp_ar.fit_my_master_piece(
            self.frame, masterpiece, self.masterpiece_scale
        )
        self.frame = ar_image

    def frame_flip(self):
        cv.flip(self.frame, 1)

    def make_frame_writable(self):
        self.frame.flags.writeable = True

    def to_gray(self):
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

    def to_rgb(self):
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)

    def to_bgr(self):
        self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2BGR)

    def rescale_frame(self, percent=75):
        width = int(self.frame.shape[1] * percent / 100)
        height = int(self.frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv.resize(self.frame, dim, interpolation=cv.INTER_AREA)

    def text_goodbye(self):
        cv.putText(
            self.frame,
            "Good Bye...",
            (10, self.frame.shape[0] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    def text_gesture(self, prediction):
        cv.putText(
            self.frame,
            prediction,
            (10, self.frame.shape[0] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    def text_screenshot(self):
        cv.putText(
            self.frame,
            "Screenshot Captured...",
            (10, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
