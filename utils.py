# %%
import matplotlib.pyplot as plt

def plot_hough_transform(plot_data, titles):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 15))
    n = len(plot_data)

    for i in range(n):
        sp = fig.add_subplot(1, n, i+1)
        mode = plot_data[i][1]
        if mode == 'h_space':
            h_space = plot_data[i][0]
            plot_hough_space(sp, h_space, titles[i])
        else:
            img = plot_data[i][0]
            if mode == 'gray':
                sp.imshow(img, cmap='gray')
            else:
                sp.imshow(img)
            sp.set_title(titles[i])

    plt.show()

def plot_hough_space(sp, h_space, title):
    for h1 in h_space[1]:
        sp.plot(h_space[0], h1, color='white', alpha=0.05)
    sp.set_title(title)

def display_single_filter(img_orig, img_filt, orig_name="Original", effect_name=None, aspect='equal', grad=None):
    if grad is None:
        n = 2
        w = 5
    else:
        n = 3
        w = 8
    fig, axes = plt.subplots(
        1, n, sharex=True, sharey=True, figsize=(8, w), dpi=100)
    # fig.suptitle(filter_name)
    axes[0].set_title(orig_name,
                      fontdict=None, loc='center', color="k")
    axes[0].imshow(img_orig, cmap='gray', aspect=aspect)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)

    axes[1].set_title(effect_name,
                      fontdict=None, loc='center', color="k")
    axes[1].imshow(img_filt, cmap='gray', aspect=aspect)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)

    if n == 3:
        axes[2].set_title('Gradient',
                          fontdict=None, loc='center', color="k")
        axes[2].imshow(grad, cmap='gray', aspect=aspect)
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)

    for ax in axes:
        ax.label_outer()
    plt.show()
# %%
