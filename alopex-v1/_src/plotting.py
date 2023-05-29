"""A plotting utilities for json formatted logs."""
from __future__ import annotations
import typing as tp
from pathlib import Path
import json

try:
    _MATPLOT_SEABORN_AVAILABLE = True
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    _MATPLOT_SEABORN_AVAILABLE = False


def plot_log_on_disk(
    file: str | Path,
    log_file: str | Path,
    y_keys: str | tp.Sequence[str],
    x_key: str = "step",
    context: str = "poster",
    style: str = "darkgrid",
    palette: str = "bright",
    labels: str | tp.Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
    dpi: int = 300,
) -> None:
    """Plot the trajectry of specified metrics using log files created by
        `alopex.DiskLogger`. Note that you must install `matplotlib` and `seaborn`
        in advance to use this method.

    Args:
        file: Filename to output the graph.
        log_file: Log file dumped by `alopex.DiskLogger`
        y_keys: Value names of y-axis to plot.
        x_key: Value name of x-axis to plot.
        context: Context parameter of seaborn.
        style: Style parameter of seaborn.
        palette: Palette parameter of seaborn.
        labels: Name of lines. If None, use `y_keys` if len(y_keys) > 1.
            If len(y_keys)==1 and not specified, do not show the label in legend.
        xlabel: Name of x-axis. If None, use `x_key`.
        ylabel: Name of y-axis. If None, use `y_keys[0]` if len(y_keys) == 1; otherwise do not show.
        title: Graph title.
        dpi: DPI parameter.
    """
    if not _MATPLOT_SEABORN_AVAILABLE:
        raise ImportError("matplotlib or seaborn is not available.")

    # canonicalize arguments.
    file_path = Path(file)
    log_file_path = Path(log_file)

    if isinstance(y_keys, str):
        y_keys = (y_keys,)

    if labels is None:
        labels = [None] if len(y_keys) == 1 else y_keys
    elif isinstance(labels, str):
        labels = (labels,)

    if xlabel is None:
        xlabel = x_key
    if ylabel is None:
        ylabel = y_keys[0] if len(y_keys) == 1 else None

    assert len(y_keys) == len(labels), "Mismatch the number of elements between y_keys and labels."
    assert xlim is None or len(xlim) == 2
    assert ylim is None or len(ylim) == 2

    # setup seaborn
    sns.set(
        context=context,
        style=style,
        palette=palette,
        rc={
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "legend.framealpha": 1,
            "legend.fontsize": 15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.titlesize": 23,
            "axes.labelsize": 20,
        },
    )

    # Read log file.
    log = json.loads(log_file_path.read_text())

    fig, ax = plt.subplots()
    for y_key, label in zip(y_keys, labels):
        x = [elem[x_key] for elem in log if x_key in elem and y_key in elem]
        y = [elem[y_key] for elem in log if x_key in elem and y_key in elem]
        if len(x) > 0 and len(y) > 0:
            ax.plot(x, y, label=label)

    if ax.has_data():
        ax.set_xlabel(xlabel=xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel=ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if title is not None:
            ax.set_title(title)

        if labels[0] is None:
            # legend is not required.
            fig.savefig(file_path, bbox_inches="tight")
        else:
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            fig.savefig(file_path, bbox_extra_artists=(legend,), bbox_inches="tight", dpi=dpi)
    else:
        raise RuntimeError("Failed to plot any specified data. Check x_key and y_keys.")

    plt.close(fig)
