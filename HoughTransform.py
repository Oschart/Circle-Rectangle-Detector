# %%
import numpy as np
import matplotlib.pyplot as plt

import cv2


class HoughTransform:

    def __init__(self):
        # Line and circle thresholds
        self.line_th = 50
        self.circle_th = 50
        self.origin_th = 5

        self.r_min = 20
        self.r_max = 100

    def apply_edge_filter(self, X):
        return cv2.Canny(X, 50, 150, apertureSize=3)

    # TODO: change in the code
    def detect_lines(self, img, th):
        # Rho and Theta ranges
        Theta = np.deg2rad(np.arange(0.0, 180.0, step=1.0))
        height, width = img.shape
        diag_len = int(
            np.ceil(np.sqrt(width * width + height * height)))   # max_dist

        # Rho codomain
        Rho_C = np.linspace(-diag_len, diag_len, int(2*diag_len))

        cos_t = np.cos(Theta)
        sin_t = np.sin(Theta)
        sin_cos_t = np.array([sin_t, cos_t], copy=False)

        y_idxs, x_idxs = np.nonzero(img)
        Y_X = np.array([y_idxs, x_idxs], copy=False).T
        Y_X -= np.array([height//2, width//2])

        # Rho range
        Rho_R = np.round(Y_X@sin_cos_t)
        m = Rho_R.shape[0]

        Acc, _, _ = np.histogram2d(
            np.tile(Theta, m), Rho_R.ravel(), bins=[Theta, Rho_C])
        Pk = np.argwhere(Acc.T >= th)
        LR = Rho_C[Pk[:, 0]]
        LT = Theta[Pk[:, 1]]

        h_space = [np.rad2deg(Theta), Rho_R]

        return Acc, LR, LT, h_space

    def detect_circles(self, img, th, region, R_bounds=None):
        (M, N) = img.shape
        if R_bounds == None:
            [r_max, r_min] = [max(M, N), 5]
        else:
            [r_max, r_min] = R_bounds

        # Accumulator
        A = np.zeros((r_max, M+2*r_max, N+2*r_max))
        B = np.zeros((r_max, M+2*r_max, N+2*r_max))

        theta = np.arange(0, 360)*np.pi/180
        C_y_x = np.argwhere(img)
        for r in range(r_min, r_max):
            # Circle template
            ctemp = np.zeros((2*(r+1), 2*(r+1)))
            R = r+1
            x = np.round(r*np.cos(theta)).astype(int)
            y = np.round(r*np.sin(theta)).astype(int)
            ctemp[R+x, R+y] = 1
            circum = np.argwhere(ctemp).shape[0]

            for x, y in C_y_x:
                X = [x-R+r_max, x+R+r_max]
                Y = [y-R+r_max, y+R+r_max]
                A[r, X[0]:X[1], Y[0]:Y[1]] += ctemp
            A[r][A[r]*r < th*circum] = 0

        # Local peak search
        for r, x, y in np.argwhere(A):
            temp = A[r-region:r+region, x-region:x+region, y-region:y+region]
            try:
                p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
            except:
                continue
            B[r+(p-region), x+(a-region), y+(b-region)] = 1

        return B[:, r_max:-r_max, r_max:-r_max]

    def draw_lines(self, img, LR, LT):
        height, width = img.shape[0], img.shape[1]
        marked_img = np.array(img, copy=True)

        for rho, theta in zip(LR, LT):
            a = np.cos(theta)
            b = np.sin(theta)
            offset = a*(width//2) + b*(height//2)
            rho += offset
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(marked_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return marked_img

    def draw_circles(self, orig_img, CS):
        img = np.array(orig_img, copy=True)
        circle_specs = np.argwhere(CS)  # Extracting the circle information
        for r, x, y in circle_specs:
            # draw the outer circle
            cv2.circle(img, (y, x), r, (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (y, x), 2, (0, 0, 255), 3)
        return img
