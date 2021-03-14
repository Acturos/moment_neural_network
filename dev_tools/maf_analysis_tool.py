import numpy as np

from mnn_core.maf import *


def show_2d_grid_map(fun, xmin=0, xmax=1, ymin=0, ymax=1, log_scale=False):
    '''Visualize a function from R2 to R2 using a grid representation'''

    num_grid_lines = 30
    num_points = 1000

    if log_scale:
        x_grid = np.logspace(xmin, xmax, num_grid_lines)
        y_grid = np.logspace(ymin, ymax, num_grid_lines)
        x = np.logspace(xmin, xmax, num_points)
        y = np.logspace(ymin, ymax, num_points)
    else:
        x_grid = np.linspace(xmin, xmax, num_grid_lines)
        y_grid = np.linspace(ymin, ymax, num_grid_lines)
        x = np.linspace(xmin, xmax, num_points)
        y = np.linspace(ymin, ymax, num_points)

    fig = plt.figure()

    # Plot the input    
    ax1 = fig.add_subplot(121)

    if log_scale:
        for yy in y_grid:
            ax1.loglog(x, yy * np.ones(x.shape), 'b', linewidth=1)
        for xx in x_grid:
            ax1.loglog(xx * np.ones(y.shape), y, 'r', linewidth=1)
    else:
        for yy in y_grid:
            ax1.plot(x, yy * np.ones(x.shape), 'b', linewidth=1)
        for xx in x_grid:
            ax1.plot(xx * np.ones(y.shape), y, 'r', linewidth=1)

    ax1.set_xlabel(r'Input mean $\bar{\mu}$', fontsize=16)
    ax1.set_ylabel(r'Input std $\bar{\sigma}$', fontsize=16)
    ax1.axis('equal')

    # Plot the output
    ax2 = fig.add_subplot(122)

    for yy in y_grid:
        x_out, y_out = fun(x, yy * np.ones(x.shape))
        ax2.plot(x_out, y_out, 'b', linewidth=1)

    for xx in x_grid:
        x_out, y_out = fun(xx * np.ones(y.shape), y)
        ax2.plot(x_out, y_out, 'r', linewidth=1)

        # plot 'boundary': fix input sigma, input mu = (-inf,+inf)
    x_boundary = np.concatenate((-np.logspace(5, -2, num_points), np.array([0.0]), np.logspace(-2, 5, num_points)))
    y_boundary = np.ones(x_boundary.shape) * ymax
    x_boundary_out, y_boundary_out = fun(x_boundary, y_boundary)
    ax2.plot(x_boundary_out, y_boundary_out, 'k')
    ax2.plot([0, 0.2], [0, 0], 'k')
    ax2.plot([0, 0.2], [0, 0.2], 'k--')

    ax2.set_xlabel('Output mean $\mu$', fontsize=16)
    ax2.set_ylabel('Output std $\sigma$', fontsize=16)
    ax2.axis('equal')
    fig.tight_layout()
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)


if __name__ == "__main__":
    maf = MomentActivation()

    fun = lambda x, y: (maf.mean(x, y), maf.std(x, y)[0])

    # show_2d_grid_map(fun, xmin = -1, xmax = 5, ymin = -1, ymax = 5, log_scale = True)
    show_2d_grid_map(fun, xmin=-2, xmax=-2 + 15, ymin=0, ymax=15, log_scale=False)
    # show_2d_grid_map(fun, xmin = -100000, xmax = 200000, ymin = 0, ymax = 1000000, log_scale = False)

# To run this file, use the command from the root directory
# runfile('./dev_tools/maf_analysis_tool.py', wdir='./')
