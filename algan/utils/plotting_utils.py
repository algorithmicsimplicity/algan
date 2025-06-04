import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Qt5Agg')


def plot_tensor(x, norm=True, save_file=None, background=None, background_cmap=None, figsize=3/100, **kwargs):
    #plot_image(tf.to_pil_image(x.squeeze().cpu()))
    if x.dim() == 2:
        x = x.unsqueeze(0)
        #s = math.isqrt(x.shape[-1])
        #x = x.view(len(x), 1, s, s)
    if x.dim() == 4:
        x = x[0]

    if norm:
        x = x - x.amin()
        x = x /  x.amax()

    #dpi = 100
    #s = 3
    figsize = figsize * x.shape[-2], figsize * x.shape[-1]

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)#(figsize[0], figsize[0]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    if background is not None:
        ax.imshow(background.permute(1,2,0).detach().cpu().squeeze(), vmin=0, vmax=1)#, cmap=background_cmap)
    x = x.permute(1, 2, 0).detach().cpu().squeeze()
    #ax.imshow(x, vmin=0, vmax=1, **kwargs, alpha=((x-0.5).abs()*2).clamp(0,1) if background is not None else 1)
    ax.imshow(x, vmin=0, vmax=1, **kwargs, alpha=0.95 if background is not None else 1)
    if save_file is None:
        plt.pause(1)
        plt.show()
    else:
        ext = 'pdf'#'png'
        f = f'{save_file}.{ext}'
        j = 0
        while os.path.exists(f):
            f = f'{save_file}_{j}.{ext}'
            j += 1
        plt.savefig(f, bbox_inches='tight',transparent=True, pad_inches=0)
        plt.clf()
        plt.close()
