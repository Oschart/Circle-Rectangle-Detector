
import numpy as np

import cv2


class HoughTransform:

    def __init__(self):
        # Line and circle thresholds
        self.line_th = 50
        self.circle_th = 50
        self.origin_th = 5

    def apply_edge_filter(self, X):
        return cv2.Canny(X, 100, 150)

    # TODO: change in the code
    def line_transform(self, X):
        # Rho and Theta ranges
        thetas = np.deg2rad(np.arange(-90.0, 90.0, step=1.0))
        width, height = X.shape
        diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
        rhos = np.linspace(-diag_len, diag_len, int(diag_len * 2.0))

        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)

        # Hough accumulator array of theta vs rho
        Acc = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(X)  # (row, col) indexes to edges

        # Vote in the hough accumulator
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idx in range(num_thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
                Acc[rho, t_idx] += 1
        return Acc, thetas, rhos

    def extract_lines(self, Acc, thetas, rhos):
        Pk = np.argwhere(Acc >= self.line_th)
        LR = rhos[Pk[:, 0]]
        LT = thetas[Pk[:, 1]]
        return LR, LT

    def extract_rects(self, LR, LT):
        rects = []
        n = LR.shape[0]
        return n
