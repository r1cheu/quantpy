import numpy as np
import pandas as pd

from quantpy import EqPopMeanEffect

LATEX_HEAD = r"""
\documentclass{article}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{graphicx}

\captionsetup{labelformat=empty,
              font={large, bf},
              aboveskip=0.5mm,
              belowskip=0.5mm}


\begin{document}
"""


def read_qtl(data_path: str, eff: dict, se: dict) -> list[str]:
    """Read the QTL data and calculate the mean effect of each loci than build the latex table.

    Args:
        data_path: str, path to the QTL data.
        a: float, total additive effect, calculated by the JointScale.
        d: float, total dominance effect, calculated by the JointScale.
        d_a: float, total dominance by additive effect, calculated by the JointScale.

    Returns:
        a list of str, contains the latex code for the QTL table.
    """
    qtl = pd.read_csv(data_path)
    qtl_list = []
    for _, row in qtl.iterrows():
        loci, A, H, B, gene = (
            row.iloc[0],
            row.iloc[1],
            row.iloc[2],
            row.iloc[3],
            row.iloc[4],
        )

        meaneff = EqPopMeanEffect(
            f=np.array([0.5, 0.5]), y=np.array([A, H, B])
        )  # calculated the a and d and d/a for each loci
        qtl_list.append(
            [
                f"QTL{loci}",
                f"{meaneff.a:.2f}",
                f"{meaneff.d:.2f}",
                f"{meaneff.d_by_a:.2f}",
                "\\textif{" + f"{gene}" + "}",
            ]
        )
    qtl_list.append(["...", "...", "...", "...", ""])

    total_line = ["Total"]
    for eff, value in eff.items():
        if value != "/" and eff != "d_a":
            total_line.append(
                f"${value:.2f}^" + "{" + f"\\pm{se[eff]:.2f}" + "}$"
            )
        else:
            total_line.append(f"{value:.2f}")
    total_line.append("")

    qtl_list.append(total_line)
    qtl_df = pd.DataFrame(
        qtl_list,
        columns=pd.Index(
            [
                "\\textbf{Loci}",
                "\\textbf{\\textit{a}}",
                "\\textbf{\\textit{d}}",
                "\\textbf{\\textit{d/a}}",
                "\\textbf{Cloned Gene}",
            ]
        ),
    )
    qtl_latex = qtl_df.to_latex(
        index=False,
        column_format="llllc",
        caption="Genetic effect",
        position="!htbp",
    ).split("\n")

    qtl_latex.insert(1, "\\centering")
    return qtl_latex
