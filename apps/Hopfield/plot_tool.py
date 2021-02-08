from hopfield import *
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2d_seq(data: list, fig=None, interval: int = 200, cmap="gray", name="mean", plot_op=False):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.1, wspace=0.1)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    frames = []

    for i in range(len(data)):
        img = torch.squeeze(data[i], dim=0).detach()
        frames.append(img)

    im0 = frames[0]
    im = ax.imshow(im0, cmap=cmap)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title(name + " Frame 0")

    def animate(idx):
        next_img = frames[idx]
        im = ax.imshow(next_img, cmap=cmap)
        cax.cla()
        fig.colorbar(im, cax=cax)
        tx.set_text(name + " Frame {:}".format(idx))

    ani = animation.FuncAnimation(fig=fig, func=animate, frames=len(data), interval=interval)
    if plot_op:
        plt.show()
    return ani


def compare_inp_and_out(inp_u: Tensor, inp_s: Tensor, inp_r: Tensor,
                        out_u: Tensor, out_s: Tensor, out_r: Tensor, cmap="gray"):
    nrows = 2
    ncols = 3
    fig = plt.figure(figsize=(ncols*4, nrows*4))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.15, wspace=0.1)
    ax = fig.add_subplot(nrows, ncols, 1)
    im = ax.imshow(torch.squeeze(inp_u, dim=0), cmap=cmap)
    plt.colorbar(im)
    ax.set_title("input mean")

    ax = fig.add_subplot(nrows, ncols, 2)
    im = ax.imshow(torch.squeeze(inp_s, dim=0), cmap=cmap)
    plt.colorbar(im)
    ax.set_title("input std")

    ax = fig.add_subplot(nrows, ncols, 3)
    im = ax.imshow(torch.squeeze(inp_r, dim=0), cmap=cmap)
    plt.colorbar(im)
    ax.set_title("input correlation")

    ax = fig.add_subplot(nrows, ncols, 4)
    im = ax.imshow(torch.squeeze(out_u, dim=0), cmap=cmap)
    plt.colorbar(im)
    ax.set_title("output mean")

    ax = fig.add_subplot(nrows, ncols, 5)
    im = ax.imshow(torch.squeeze(out_s, dim=0), cmap=cmap)
    plt.colorbar(im)
    ax.set_title("output std")

    ax = fig.add_subplot(nrows, ncols, 6)
    im = ax.imshow(torch.squeeze(out_r, dim=0), cmap=cmap)
    plt.colorbar(im)
    ax.set_title("output correlation")

    return fig


if __name__ == "__main__":
    net = Net(16, 16, 50, 10, True)
    state = torch.load("hopfield_demo.pt")
    net.load_state_dict(state)
    net.eval()
    path = "BioID_0010.pgm"
    tool = PrepareData(16)
    pic = tool.read_img(path)
    pic.show()
    u, s, r, tu, ts, tr = tool.fetch_train_data(path, 1)

    with torch.no_grad():
        ru, rs, rr = net(tu, ts, tr)

    compare_inp_and_out(tu, ts, tr, ru[-1], rs[-1], rr[-1])
    plt.show()


