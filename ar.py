import cv2 as cv
import numpy as np


class MasterPiece:
    def __init__(self) -> None:
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        self.parameters = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)

    def fit_my_master_piece(self, frame, master_piece):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
        scale_percent = 100  # (1000 / frame.shape[0]) * 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        width2 = int(master_piece.shape[1] * scale_percent / 100)
        height2 = int(master_piece.shape[0] * scale_percent / 100)
        dim = (width2, height2)
        master_piece = cv.resize(master_piece, dim, interpolation=cv.INTER_AREA)

        warped = self.find_and_warp(frame, master_piece, cornerIds=(1, 2, 4, 3))
        if warped is not None:
            frame = warped
        return frame

    def find_and_warp(self, frame, source, cornerIds):
        (imgH, imgW) = frame.shape[:2]
        (srcH, srcW) = source.shape[:2]
        markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(
            frame
        )
        if len(markerCorners) != 4:
            markerIds = np.array([])
        else:
            markerIds.flatten()
        refPts = []
        for i in cornerIds:
            j = np.squeeze(np.where(markerIds == i))
            if j.size == 0:
                continue
            else:
                j = j[0]

            markerCorners = np.array(markerCorners)
            corner = np.squeeze(markerCorners[j])
            refPts.append(corner)
        if len(refPts) != 4:
            return None
        (refPtTL, refPtTR, refPtBR, refPtBL) = np.array(refPts)
        dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
        dstMat = np.array(dstMat)
        srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
        (H, _) = cv.findHomography(srcMat, dstMat)
        warped = cv.warpPerspective(source, H, (imgW, imgH))
        mask = np.zeros((imgH, imgW), dtype="uint8")
        cv.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv.LINE_AA)
        maskScaled = mask.copy() / 255.0
        maskScaled = np.dstack([maskScaled] * 3)
        warpedMultiplied = cv.multiply(warped.astype("float"), maskScaled)
        imageMultiplied = cv.multiply(frame.astype(float), 1.0 - maskScaled)
        output = cv.add(warpedMultiplied, imageMultiplied)
        output = output.astype("uint8")
        return output
