from typing import List

import matplotlib.pyplot as plt
import numpy as np


def pie(data: np.ndarray, label: List[str], population: str, save_path: str):
    """Plot pie chart for genetic variance decomposition.

    Args:
        data:  The values contained in the pie chart. Should be a 1D numpy array contains
        the variance for each component.
        label: name for each component. e.g. ["[a]", "[d]", "others"]
        population: name for population, which will be displayed in the title of the plot.
        save_path: path to save the plot.

    Raises:
        ValueError: if the label is not in the list ["[a]", "[d]", "[aa]", "[ad]", "[dd]", "others"]

    """
    _, ax = plt.subplots()
    patches, texts, pcts = ax.pie(
        x=data,
        labels=label,
        autopct="%.1f%%",
        textprops={"size": "x-large"},
        startangle=-90,
    )

    for i, patch in enumerate(patches):
        if texts[i].get_text() == "[a]":
            patch.set_facecolor("#FFD966")
        elif texts[i].get_text() == "[d]":
            patch.set_facecolor("#F57A27")
        elif texts[i].get_text() == "others":
            patch.set_facecolor("grey")
        elif texts[i].get_text() == "[aa]":
            patch.set_facecolor("#A9D18E")
        elif texts[i].get_text() == "[ad]":
            patch.set_facecolor("#5F9FDB")
        elif texts[i].get_text() == "[dd]":
            patch.set_facecolor("#BFBFBF")
        else:
            raise ValueError("Unknown text")

        texts[i].set_color(patch.get_facecolor())
        if float(pcts[i].get_text()[:-1]) < 10:  # Hide small percentage
            pcts[i].set_text("")

    plt.setp(pcts, color="k", fontweight=600, fontsize=20)
    plt.setp(texts, fontweight=600, fontsize=22)
    ax.set_title(
        f"Genetic variance decomposition\n{population}",
        fontsize=18,
        fontweight=600,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
