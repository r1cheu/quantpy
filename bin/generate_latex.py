#! /usr/bin/env python3
import json
import os
from argparse import ArgumentParser

from quantpy import JointScale, VarianceDecomposition, pie
from quantpy.latex import LATEX_HEAD, read_qtl


def get_args():
    parser = ArgumentParser(
        description="Generate LaTeX code for Joint Scale Analysis And Variance Decomposition"
    )
    parser.add_argument(
        "json",
        type=str,
        help="the json file containing the data, see the example file for the format",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="the output zip, default is %(default)s",
        default="output.zip",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_false",
        help="show the analysis results in the terminal, default is %(default)s",
    )

    return parser.parse_args()


def pie_plot(variance: dict, population, trait) -> str:
    title = f"{trait} {population}"
    os.makedirs("img", exist_ok=True)
    img_path = f"img/{trait}_{population}.png"

    var, label = [], []
    for k, v in variance.items():
        if k == "SSY":
            continue
        var.append(v)
        label.append(f"[{k}]")
    label.append("others")
    var.append(variance["SSY"] - sum(var))

    pie(var, label, title, img_path)

    return img_path


if __name__ == "__main__":
    args = get_args()

    with open(args.json) as f:
        data = json.load(f)
    print(f"The json file you provided contains {len(data)} entries")

    latex_text = [LATEX_HEAD]

    for entry in data:
        population = f"{entry["parent1"]} $\\times$ {entry["parent2"]}"

        js = JointScale(entry["mean_var_file"], entry["trait"], population)
        js.fit(entry["js_effects"], print_result=args.verbose)
        latex_text.extend(js.latex)

        latex_text.extend(read_qtl(entry["QTL_file"], js.a_d_value, js.eff_se))

        vd = VarianceDecomposition(entry["mean_var_file"])
        vd.fit(entry["vd_effects"], print_result=args.verbose)

        variance = {}
        for e in entry["js_effects"] + ["SSY"]:
            variance[e] = vd.variance[e]

        img_path = pie_plot(
            variance, population.replace(" $\\times$ ", "x"), entry["trait"]
        )

        latex_text.extend(
            [
                "\\begin{figure}[!hbtp]",
                "\\centerline{\\includegraphics[width=0.7\\linewidth]{"
                + img_path
                + "}}",
                "\\end{figure}",
            ]
        )
        latex_text.append("\\newpage")

    latex_text.append("\\end{document}")

    with open("main.tex", "w") as f:
        f.write("\n".join(latex_text))

    os.system(f"zip -r -q {args.output} img main.tex")
    os.system("rm -rf img main.tex")
