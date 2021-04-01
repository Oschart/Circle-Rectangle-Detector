# %%
import matplotlib.pyplot as plt
from math import ceil
def plot_showcase(plot_data, figsize=(15,13)):
    n = len(plot_data)
    nr = ceil(n/2)
    plt.style.use('dark_background')
    fig, axs = plt.subplots(nr, 2, figsize=figsize, squeeze=False)

    for i in range(n):
        ii, jj = i//2, i%2
        sp = axs[ii, jj]
        mode = plot_data[i][1]
        title = plot_data[i][2]
        if mode == 'h_space':
            h_space = plot_data[i][0]
            plot_hough_space(sp, h_space, title)
        else:
            img = plot_data[i][0]
            if mode == 'gray':
                sp.imshow(img, cmap='gray')
            else:
                sp.imshow(img)
            sp.set_title(title)

    plt.show()

def plot_hough_space(sp, h_space, title):
    for h1 in h_space[1]:
        sp.plot(h_space[0], h1, color='white', alpha=0.01)
    sp.set_title(title)
    sp.set_xlabel('Theta (deg)')
    sp.set_ylabel('Rho')

