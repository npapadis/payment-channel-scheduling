import matplotlib.pyplot as plt
from pathlib import Path


def save_legend(fig, lines, labels, legend, legend_directory, legend_filename):
    # Code modified from https://gist.github.com/rldotai/012dbb3294bb2599ff82e61e82356990

    # ---------------------------------------------------------------------
    # Create a separate figure for the legend
    # ---------------------------------------------------------------------
    # Bounding box for legend (in order to get proportions right)
    # Issuing a draw command seems necessary in order to ensure the layout
    # happens correctly; I was getting weird bounding boxes without it.
    # fig.canvas.draw()
    # This gives pixel coordinates for the legend bbox (although perhaps
    # if you are using a different renderer you might get something else).
    legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
    # Convert pixel coordinates to inches
    legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())

    # Create the separate figure, with appropriate width and height
    # Create a separate figure for the legend
    legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))

    # Recreate the legend on the separate figure/axis
    legend_squared = legend_ax.legend(
        # *ax.get_legend_handles_labels(),
        lines,
        labels,
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=legend_fig.transFigure,
        frameon=False,
        fancybox=None,
        shadow=False,
        # ncol=legend_cols,
        mode='expand',
    )

    # Remove everything else from the legend's figure
    legend_ax.axis('off')

    # Save the legend as a separate figure
    # print(f"Saving to: {legend_fullpath}")

    Path(legend_directory).mkdir(parents=True, exist_ok=True)
    legend_fullpath = legend_directory + "/" + legend_filename
    legend_fig.savefig(
        legend_fullpath,
        bbox_inches='tight',
        bbox_extra_artists=[legend_squared],
    )
