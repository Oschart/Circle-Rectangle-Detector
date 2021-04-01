# %%
import matplotlib.pyplot as plt
import numpy as np
#import cv2
from skimage.io import imread, imshow


from HoughTransform import HoughTransform
import cv2
from utils import plot_showcase

LINE_THRESHOLD = 92     # Line detection threshold
CIRC_THRESHOLD = 13     # Circle detection threshold

def run(case_ids=range(4)):

    test_cases = [f'./test_cases/case{i}.jpg' for i in case_ids]
    for test_img in test_cases:

        init_final, transform_imgs = process_img(test_img)
        orig_img, final_img = init_final
        edge_img, h_space, imcr = transform_imgs

        plot_data1 = [(orig_img, None, 'Original Image'),
                      (edge_img, 'gray', 'Edge Image'),
                      (h_space, 'h_space', 'Hough Space'),
                      (imcr, None, 'Detected Shapes')]
        plot_data2 = [(orig_img, None, 'Original Image'),
                      (final_img, None, 'Annotated Image')]

        plot_showcase(plot_data1, figsize=(12, 9))
        plot_showcase(plot_data2, figsize=(8, 8))
        # %%


def process_img(img_path):
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 11)

    HT = HoughTransform()

    edge_img = HT.apply_edge_filter(gray_img)

    CS = HT.detect_circles(edge_img, CIRC_THRESHOLD, 15, R_bounds=[30, 22])
    grad_img_rgb = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
    imc = HT.draw_circles(grad_img_rgb, CS)

    Acc, LR, LT, h_space = HT.detect_lines(edge_img, th=LINE_THRESHOLD)

    lines = list(zip(LR, LT))
    ulines = HT.clear_similar_lines(lines)

    imcl = HT.draw_lines(imc, ulines)

    rects = HT.form_rects(ulines)
    imcr = HT.draw_rects(imc, rects)

    final_img = HT.draw_circles(HT.draw_rects(orig_img, rects), CS)

    return [orig_img, final_img], [edge_img, h_space, imcr]

