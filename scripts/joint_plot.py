import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.datamodules.helper_data_classes import SampleInformation

sns.set_theme(style="whitegrid", font_scale=1.75)
sns.set_style("ticks")
sns.set_palette("flare")

x_col_name = "score"
y_col_name = "utility"

colorbar = True  # ToDo: Add support
only_normalized = False


# ~~~~~~
# Helpers for generating a debug dataset
def get_normal_random_data(mean, scale, num_samples, seed=123):
    np.random.seed(seed)
    std_norm_data = np.random.randn(num_samples)

    data = std_norm_data * scale + mean
    return data


def get_si_data(dim1, dim2):
    dim1 = get_normal_random_data(**dim1)
    dim2 = get_normal_random_data(**dim2)
    data = [SampleInformation(x1, x2).get_summary() for x1, x2 in zip(dim1, dim2)]

    return data


# ~~~~~~


def get_filter_summary_df(si_iterable, only_normalized):
    df_data = []
    for si in si_iterable:
        if si["score"][0] is None or (only_normalized and not si["score"][1]):
            # Filter elements
            # - without a valid score; or
            # - cannot be normalized (and we are plotting normalized data)
            continue
        si["score"] = si["score"][0]
        df_data.append(si)

    return pd.DataFrame(df_data)


# ~~~~~~
# Helpers for generating the marginal plots
def _hist_marg(_, df, x_col_name, y_col_name, vertical=None, labels=None, **kwargs):
    """Wrapper to call hist_marg() from seaborn plot_marginals()."""
    assert vertical in set([True, False])

    col_name = y_col_name if vertical else x_col_name

    if labels is not None:
        kwargs["label"] = labels["y"] if vertical else labels["x"]
    hist_marg(df, col_name, vertical=vertical, **kwargs)


def hist_marg(df, col_name, vertical=False, ax=None, **kwargs):
    assert col_name in df

    # Get current axis
    if ax is None:
        ax = plt.gca()

    xlabel, ylabel = ax.set_xlabel, ax.set_ylabel
    xlim, ylim = ax.set_xlim, ax.set_ylim

    data_key = "x"
    if vertical:
        data_key = "y"
        xlabel, ylabel = ylabel, xlabel
        xlim, ylim = ylim, xlim  # ToDo: Set the limits across all plots in a meaningful way

    kwargs[data_key] = df[col_name]
    if "extent" in kwargs:
        extent = kwargs.pop("extent")
        if col_name == "utility":
            kwargs["binrange"] = (extent[2], extent[3])
        elif col_name == "score":
            kwargs["binrange"] = (extent[0], extent[1])

    label = kwargs.pop("label", None)
    sns.histplot(**kwargs)
    if kwargs.get("kde", False):
        # Change the properties of the KDE line
        ax.lines[0].set_color("black")

    if label is not None:
        ylabel(label)


# ~~~~~~


def get_plot(
    df,
    title=None,
    difference=True,
    ratio=False,
    only_normalized=True,
    grid_kwargs=None,
    kind="hex",
    joint_kwargs={},
    marginal_kwargs={},
    show_plot=True,
    close_plot=True,
    save_to_file=None,
    colorbar=False,
    plot_correlation=False,
    xlabel=None,
    ylabel=None,
    utility_max=None,
    single_line=True
):
    if df is None:
        # Get a debugging dataset
        num_samples = 500
        si_iterable = get_si_data(
            {"mean": 10, "scale": 5, "num_samples": num_samples, "seed": 123},
            {"mean": 3, "scale": 1, "num_samples": num_samples, "seed": 12345},
        )
        df = get_filter_summary_df(si_iterable, only_normalized)

    assert len(df) != 0

    x_col_name = "score"
    y_col_name = "utility"
    g = sns.JointGrid(data=df, x=x_col_name, y=y_col_name, **grid_kwargs)

    if single_line:
        g.figure.set_figheight(6.5)
    elif xlabel:
        g.figure.set_figheight(6.43)
    elif colorbar:
        g.figure.set_figheight(6.61)

    if kind.startswith("hex"):
        g.plot_joint(plt.hexbin, **joint_kwargs)
    else:
        # We could support more types of plots for the joint distribution if necessary (e.g. heat map)
        raise NotImplemented()

    g.plot_marginals(
        _hist_marg,
        df=df,
        x_col_name=x_col_name,
        y_col_name=y_col_name,
        labels={"x": "Sample Count", "y": "Sample Count"},
        **marginal_kwargs,
    )

    if xlabel:
        g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
    else:
        g.set_axis_labels(ylabel=ylabel)

    # Show grid in marginal plots
    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)

    # Add line for the score where the target prediction would lie (i.e. 0)
    if difference:
        g.refline(x=0, linestyle="dashed", color="crimson")

    # Add line for perfect utility
    g.refline(y=1, linestyle='dashed', color='crimson')

    if colorbar:
        label = "Correlation" if plot_correlation else "Count"
        cb = plt.colorbar(plt.gci(), label=label, ax=g.ax_marg_x, use_gridspec=True, location="top")
        cb.outline.set_visible(False)

    if title:
        g.figure.suptitle(title)

    if utility_max:
        plt.xlim(0, utility_max)

    if save_to_file:
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        plt.savefig(f"{save_to_file}.pdf", bbox_inches='tight')
        # plt.savefig(f"{save_to_file}.png", dpi=300)
    if show_plot:
        plt.show()

    if close_plot:
        plt.close()


if __name__ == "__main__":
    grid_kwargs = {"ratio": 3, "space": 0.2, "marginal_ticks": True}
    joint_kwargs = {
        "gridsize": 20,
        "mincnt": 1,
        "cmap": "flare",
    }
    marginal_kwargs = {"edgecolor": "black", "bins": 40, "linewidth": 0.15, "alpha": 0.8, "kde": True}
    get_plot(
        None,
        title="Dummy data plot",
        only_normalized=False,
        grid_kwargs=grid_kwargs,
        colorbar=True,
        joint_kwargs=joint_kwargs,
        marginal_kwargs=marginal_kwargs,
    )
