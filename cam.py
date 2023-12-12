import cv2 as cv


# interaction with webcam device is made through this class methods
class CameraProcessor:
    # initialize the media pipe ar and default webcam params
    def __init__(self, mp_ar) -> None:
        self.mp_ar = mp_ar
        self.masterpiece_scale = 100

    # start capturing frames of the webcam input
    def start(self):
        self.cap = cv.VideoCapture(0)

    # stop capturing of webcam
    def stop(self):
        self.cap.release()

    # returns true if the webcam is capturing frames,
    # it should be used in an infinite loop to get the frames constantly
    def is_capturing(self):
        return self.cap.isOpened()

    # read the latest frame of the webcam capturing
    # it sets private frame variable of the class
    def frame_read(self):
        ret, self.frame = self.cap.read()

    # return the latest captured frame
    def get_frame(self):
        return self.frame

    # replace the frame with a custom frame
    def set_frame(self, frame):
        self.frame = frame

    # sets the drawing with the screenshot of the current frame
    def set_masterpiece(self, masterpiece):
        ar_image = self.mp_ar.fit_my_master_piece(
            self.frame, masterpiece, self.masterpiece_scale
        )
        self.frame = ar_image

    # flips the current frame, every time it is called, the current frame flips
    def frame_flip(self):
        cv.flip(self.frame, 1)

    # sets the current frame as a writable frame
    def make_frame_writable(self):
        self.frame.flags.writeable = True

    # converts the current frame to black and white
    def to_gray(self):
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

    # converts frame's bgr color set to rgb
    def to_rgb(self):
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)

    # converts frame's rgb color set to bgr
    def to_bgr(self):
        self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2BGR)

    # resize the frame according to the input percentage
    def rescale_frame(self, percent=75):
        width = int(self.frame.shape[1] * percent / 100)
        height = int(self.frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv.resize(self.frame, dim, interpolation=cv.INTER_AREA)

    # adds goodby text into the frame
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

    # adds current detected gesture and precision into the frame
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

    # adds screenshot capturing status into the frame
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
