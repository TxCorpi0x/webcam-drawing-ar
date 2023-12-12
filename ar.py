import cv2 as cv
import numpy as np


# every ar processing over the finger drawing is done by this class methods
class MasterPiece:
    # initialize ArUco marker detector and parameters
    def __init__(self) -> None:
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        self.parameters = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)

    # rescale the masterpiece to fit in the wrapped ArUco markers
    def fit_my_master_piece(self, frame, masterpiece, scale_percent):
        if scale_percent != 100:
            # extract current frame size and dimensions
            w = int(frame.shape[1] * scale_percent / 100)
            h = int(frame.shape[0] * scale_percent / 100)
            dim = (w, h)

            # resize the frame according to the scale percentage and calculate the
            # size and dimension after scaling
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            w2 = int(masterpiece.shape[1] * scale_percent / 100)
            h2 = int(masterpiece.shape[0] * scale_percent / 100)
            dim = (w2, h2)

            # resize the masterpiece according to the scaled dimensions of the frame
            # to preserve aspect ratio
            masterpiece = cv.resize(masterpiece, dim, interpolation=cv.INTER_AREA)

        warped = self.wrap_within_markers(frame, masterpiece, corner_ids=(1, 2, 4, 3))
        if warped is not None:
            frame = warped
        return frame

    # fits the masterpiece image aligned within the ArUco markers
    def wrap_within_markers(self, frame, masterpiece, corner_ids):
        # extract dimensions of the frame and masterpiece
        (img_h, img_w) = frame.shape[:2]
        (src_h, src_w) = masterpiece.shape[:2]

        # detect markers within the frame
        marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(
            frame
        )

        # initialize marker ids list using numpy
        if len(marker_corners) != 4:
            # not enough markers so empty array of ids
            marker_ids = np.array([])
        else:
            marker_ids.flatten()

        ref_pts = []
        for i in corner_ids:
            # remove this corner's single-dimensional items from marker ids
            j = np.squeeze(np.where(marker_ids == i))

            # if no multi dimensional marker ids found, go to the next corner
            if j.size == 0:
                continue
            else:
                j = j[0]

            # create a numpy array from marker corners
            marker_corners = np.array(marker_corners)
            # remove single-dimensional entries from corners array
            corner = np.squeeze(marker_corners[j])
            # update global reference points
            ref_pts.append(corner)

        # not enough reference points, so the marker detection did not find any proper result
        if len(ref_pts) != 4:
            return None

        # extract 4 corners of the reference points array
        (ref_pt_tl, ref_pt_tr, ref_pt_br, ref_pt_bl) = np.array(ref_pts)

        dst_mat = [ref_pt_tl[0], ref_pt_tr[1], ref_pt_br[2], ref_pt_bl[3]]
        dst_mat = np.array(dst_mat)

        src_mat = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

        # find homography and wrap the calculated perspective height
        (H, _) = cv.findHomography(src_mat, dst_mat)
        warped = cv.warpPerspective(masterpiece, H, (img_w, img_h))

        # create a mask and fill the polygon with color
        mask = np.zeros((img_h, img_w), dtype="uint8")
        cv.fillConvexPoly(mask, dst_mat.astype("int32"), (255, 255, 255), cv.LINE_AA)

        # get scaled mask and convert it to a two-dimensional array
        mask_scaled = mask.copy() / 255.0
        mask_scaled = np.dstack([mask_scaled] * 3)

        # fit the wrapped mask within the frame
        warped_multiplied = cv.multiply(warped.astype("float"), mask_scaled)
        frame_multiplied = cv.multiply(frame.astype(float), 1.0 - mask_scaled)
        output = cv.add(warped_multiplied, frame_multiplied)
        output = output.astype("uint8")
        return output
