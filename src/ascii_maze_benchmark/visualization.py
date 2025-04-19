"""Matplotlib helper for visualising benchmark results.

The benchmark runner prints a textual comparison table; this module provides
``plot_comparison_heatmap`` which turns the same data structure into a
heat‑map.  It can be used both interactively (default) and in non‑interactive
"batch" mode where a PNG is generated without requiring a display server – the
latter is handy for CI pipelines and remote servers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterable, Mapping

import numpy as np


# NOTE:
# We *deliberately* do **not** import ``matplotlib.pyplot`` at module import
# time because the backend selection (interactive window vs. head‑less *Agg*)
# must happen *before* the first pyplot import.  Instead we perform the import
# inside ``plot_comparison_heatmap`` after deciding which backend is desired.


def _ensure_sorted_sizes(sizes: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return *sizes* as a consistently sorted list (ascending width, height)."""

    return sorted(set(sizes), key=lambda s: (s[0], s[1]))


def _calculate_row(
    size_summary: Mapping, ordered_sizes: list[tuple[int, int]]
) -> tuple[list[float], list[str]]:
    """Calculate a row of data and labels based on size summaries."""
    row: list[float] = []
    lab_row: list[str] = []
    for size in ordered_sizes:
        matches, total = size_summary.get(size, (0, 0))
        if total > 0:
            pct = (matches / total) * 100.0
            row.append(pct)
            lab_row.append(f"{matches}/{total}\n({pct:.0f}%)")
        else:
            row.append(np.nan)
            lab_row.append("N/A")
    return row, lab_row


def plot_comparison_heatmap(
    summaries: Sequence[Mapping[str, Any]],
    sizes: Iterable[tuple[int, int]],
    *,
    output: str | None = None,
    interactive: bool = True,
):
    """Create a heat‑map showing exact‑match success rates.

    Parameters
    ----------
    summaries
        List as returned by the benchmark CLI (``all_summaries``).
    sizes
        Iterable of maze sizes *(w, h)* defining the column order.
    output
        If given, save the figure at this path via ``fig.savefig``.
    interactive
        When *True* (default) the figure is displayed via ``plt.show()``; when
        *False* no GUI backend is required which makes the function usable in
        headless environments.
    """

    # ------------------------------------------------------------------
    # Choose backend *before* importing pyplot.
    # ------------------------------------------------------------------

    import importlib

    import matplotlib  # type: ignore

    if not interactive:
        matplotlib.use("Agg")  # Non‑interactive, file‑only

    plt = importlib.import_module("matplotlib.pyplot")  # type: ignore
    from matplotlib.colors import ListedColormap  # Import here for the custom colormap

    # ------------------------------------------------------------------
    # Build data matrix
    # ------------------------------------------------------------------

    ordered_sizes = _ensure_sorted_sizes(sizes)

    data: list[list[float]] = []
    labels: list[list[str]] = []

    for summ in summaries:
        row_data, row_labels = _calculate_row(
            summ.get("size_summary", {}), ordered_sizes
        )
        data.append(row_data)
        labels.append(row_labels)

    data_arr = np.array(data)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(1.8 * len(ordered_sizes), 0.8 * len(summaries) + 1))

    # Create a lighter version of the RdYlGn colormap for better readability
    cmap_original = plt.cm.RdYlGn
    lighter_cmap = cmap_original(
        np.linspace(0.2, 0.8, 256)
    )  # Adjusted to include a lighter red at the low end
    lighter_cmap[0] = np.array(
        [1.0, 0.6, 0.6, 1.0]
    )  # Set the first color to ff9999 (normalized RGBA)
    cmap = ListedColormap(lighter_cmap)
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(data_arr, cmap=cmap, vmin=0, vmax=100)

    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[1]):
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                color="black" if not np.isnan(data_arr[i, j]) else "#666666",
                fontsize=9,
            )

    ax.set_xticks(
        range(len(ordered_sizes)), labels=[f"{w}x{h}" for w, h in ordered_sizes]
    )
    ax.set_yticks(range(len(summaries)), labels=[s["model_id"] for s in summaries])

    ax.set_xlabel("Maze size (WxH)")
    ax.set_ylabel("Model")
    ax.set_title("Benchmark – Exact match success rate")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate (%)")

    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=150)

    if interactive:
        plt.show(block=True)

    return fig
