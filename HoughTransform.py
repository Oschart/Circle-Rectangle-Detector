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
        return cv2.Canny(X, 100, 150, apertureSize=3)

    # TODO: change in the code
    def detect_lines(self, img, th):
        # Rho and Theta ranges
        Theta = np.deg2rad(np.arange(0.0, 180.0, step=0.5))
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
            R_max = np.max((M, N))
            R_min = 3
        else:
            [R_max, R_min] = R_bounds

        R = R_max - R_min
        # Initializing accumulator array.
        # Accumulator array is a 3 dimensional array with the dimensions representing
        # the radius, X coordinate and Y coordinate resectively.
        # Also appending a padding of 2 times R_max to overcome the problems of overflow
        A = np.zeros((R_max, M+2*R_max, N+2*R_max))
        B = np.zeros((R_max, M+2*R_max, N+2*R_max))

        # Precomputing all angles to increase the speed of the algorithm
        theta = np.arange(0, 360)*np.pi/180
        edges = np.argwhere(img)  # Extracting all edge coordinates
        for val in range(R):
            r = R_min+val
            # Creating a Circle Blueprint
            bprint = np.zeros((2*(r+1), 2*(r+1)))
            (m, n) = (r+1, r+1)  # Finding out the center of the blueprint
            for angle in theta:
                x = int(np.round(r*np.cos(angle)))
                y = int(np.round(r*np.sin(angle)))
                bprint[m+x, n+y] = 1
            constant = np.argwhere(bprint).shape[0]
            for x, y in edges:  # For each edge coordinates
                # Centering the blueprint circle over the edges
                # and updating the accumulator array
                X = [x-m+R_max, x+m+R_max]  # Computing the extreme X values
                Y = [y-n+R_max, y+n+R_max]  # Computing the extreme Y values
                A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
            A[r][A[r] < th*constant/r] = 0

        for r, x, y in np.argwhere(A):
            temp = A[r-region:r+region, x-region:x+region, y-region:y+region]
            try:
                p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
            except:
                continue
            B[r+(p-region), x+(a-region), y+(b-region)] = 1

        return B[:, R_max:-R_max, R_max:-R_max]


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

            cv2.line(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
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

