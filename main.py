# %%
import matplotlib.pyplot as plt
import numpy as np
#import cv2
from skimage.io import imread, imshow


from HoughTransform import HoughTransform
import cv2
from utils import plot_hough_transform



orig_img = cv2.imread('./test_cases/case1.jpeg')
orig_img = cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB)


img_gray = cv2.cvtColor(orig_img,cv2.COLOR_BGR2GRAY)
img_gray = cv2.bilateralFilter(img_gray,9,75,75)

HT = HoughTransform()

grad_img = HT.apply_edge_filter(img_gray)
plt.imshow(grad_img, cmap='gray')
plt.show()


'''
CS = HT.circle_transform(grad_img, 15, 15, R_bounds=[40,20])
grad_img_rgb = cv2.cvtColor(grad_img, cv2.COLOR_GRAY2RGB)
marked_img1 = HT.draw_circles(grad_img_rgb, CS)
plt.imshow(marked_img1)
plt.show()

exit()
'''

Acc, LR, LT, h_space  = HT.detect_lines(grad_img, th=170)
print(np.max(Acc))
print(LR)
print(LT)


grad_img_rgb = cv2.cvtColor(grad_img, cv2.COLOR_GRAY2RGB)
marked_img = HT.draw_lines(grad_img_rgb, LR, LT)
plt.imshow(marked_img)
plt.show()

plot_data = [(grad_img, 'gray'), (h_space, 'h_space'), (marked_img, None)]
titles = ['1', '2', '3']

plot_hough_transform(plot_data, titles)
# %%
