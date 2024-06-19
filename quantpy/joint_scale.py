from typing import List, Tuple

import numpy as np
import pandas as pd

from quantpy.stats import chi2_test, f_test, t_test


class WeightedLeastSquare:
    DM = pd.DataFrame(
        [
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5, 0.0, 0.0, 0.25],
            [1.0, 0.5, 0.5, 0.25, 0.25, 0.25],
            [1.0, -0.5, 0.5, 0.25, -0.25, 0.25],
        ],
        index=pd.Index(["p1", "p2", "f1", "f2", "b1l", "b1s"]),
        columns=pd.Index(["m", "a", "d", "aa", "ad", "dd"]),
    )

    def __init__(self, input_data: str) -> None:
        self.data = pd.read_csv(input_data, index_col=0, sep="\t")
        self._precision_matrix = np.diag(1 / self.data["var"].to_numpy())
        self._y = self.data["y"].to_numpy()
        self._X = None

    def _check_effect(self, effect, add_m=True):
        """
        Checks the effect and returns the effect list.

        Args:
            effect (list): List of effects.
            add_m (bool, optional): Whether to add 'm' to the effect list.
            Defaults to True.

        Returns:
            list: Effect list.
        """
        effect = list(effect)
        if ("m" not in effect) and add_m:
            effect.insert(0, "m")
        if not set(effect).issubset(set(self.DM.columns)):
            raise ValueError(
                "Invalid effect, only m, a, d, aa, ad, " "dd are allowed."
            )
        if len(effect) >= self._y.shape[0]:
            raise ValueError(
                "The number of effect should be less than "
                f"the number of population. but got {len(effect)}"
                f" effect and {self._y.shape[0]} population."
            )
        return effect

    def _fit(self, effect):
        self._X = self.DM.loc[self.data.index, effect].to_numpy()
        xtlx_inv = np.linalg.solve(
            self._X.T @ self._precision_matrix @ self._X,
            np.eye(self._X.shape[1]),
        )
        beta = xtlx_inv @ self._X.T @ self._precision_matrix @ self._y

        return xtlx_inv, beta


class JointScale(WeightedLeastSquare):
    def __init__(
        self,
        input_data: str,
        phenotype: str = "phenotype",
        population: str = "population",
    ) -> None:
        super().__init__(input_data)
        self.phenotype = phenotype
        self.population = population
        self._effect = []
        self._result = None

    def fit(self, effect, print_result=True):
        self._effect = self._check_effect(effect)
        xtlx_inv, beta = self._fit(self._effect)
        sig_info = self._param_test(beta, xtlx_inv)
        self._result = self._result_table(sig_info)
        if print_result:
            self._print_result()

    def cal_t(self, beta, xtlx_inv):
        """
        Calculates the t-value, degrees of freedom, and standard error.

        Args:
            beta (np.ndarray): Beta.
            xtlx_inv (np.ndarray): Inverse of Xt lambda X.

        Returns:
            Tuple[np.ndarray, int, np.ndarray]: t-value, degrees of freedom,
            and standard error.
        """
        df = self._y.shape[0] - self._X.shape[1]
        mse = (
            self._y.T @ self._precision_matrix @ self._y
            - beta.T @ self._X.T @ self._precision_matrix @ self._y
        ) / df
        se = np.sqrt(np.diag(xtlx_inv) * mse)
        t_value = beta / se
        return t_value, df, se

    def cal_chi(
        self,
        beta,
    ) -> float:
        """
        Calculates the chi-square value.

        Args:
            beta (np.ndarray): Beta.

        Returns:
            float: Chi-square value.
        """
        return np.sum(
            (self._y - self._X @ beta) ** 2 * np.diag(self._precision_matrix)
        )

    def _param_test(self, beta, xtlx_inv) -> List[Tuple[float, str, float]]:
        """
        Performs parameter test and returns the test results.

        Args:
            beta (np.ndarray): Beta.
            xtlx_inv (np.ndarray): Inverse of Xt lambda X.

        Returns:
            List[str]: Test results.
        """
        t_value, df, se = self.cal_t(beta, xtlx_inv)
        chi_value = self.cal_chi(beta)

        t_sig = list(map(t_test, t_value, [df] * len(t_value)))
        chi_sig = chi2_test(chi_value, df)

        sig_info = [
            (float(v), sig, float(s)) for v, sig, s in zip(beta, t_sig, se)
        ]
        sig_info.append((float(chi_value), chi_sig, 0.0))

        return sig_info

    @property
    def result(self) -> pd.DataFrame:
        if self._result is None:
            raise ValueError("Please fit the model first.")
        return self._result

    @property
    def effect(self) -> List[str]:
        if len(self._effect) == 0:
            raise ValueError("Please fit the model first.")
        return self._effect

    def _result_table(self, sig_info: List[Tuple[float, str, float]]):
        effect = self.effect.copy()
        effect.append("chi2")
        index = pd.Index(effect, name="effect")
        columns = pd.Index(["beta", "sig", "se"])

        return pd.DataFrame(sig_info, columns=columns, index=index)

    def _print_result(self):
        model_str = "m+[" + "]+[".join(self.result.index[1:-1]) + "]"
        lines = [("", model_str)]

        for line in self.result.itertuples():
            string = f"{line.beta:.4f}{line.sig} \u00b1{line.se:.4f}"
            lines.append((str(line.Index), string))

        print_list(lines)

    def _latex_header(self):
        header_str = f"{self.phenotype}_i = m + "
        eff = [
            "X_{" + str(idx) + "i}" + f"[{eff}]"
            for idx, eff in enumerate(self.result.index[1:-1], start=1)
        ]
        return (
            f"{self.population}"
            + "\\\\"
            + "$"
            + header_str
            + " + ".join(eff)
            + "$"
        )

    def _latex_table(self):
        lines = []
        df = 6 - len(self.effect)
        for line in self.result.itertuples():
            eff = line.Index
            string = (
                f"${line.beta:.3f}^"
                + "{"
                + f"{line.sig}"
                + "}"
                + f"\\pm {line.se:.3f}$"
            )
            if eff != "m":
                eff = f"$[{eff}]$"

            if eff == "$[chi2]$":
                eff = "$\\chi^2(df)$"
                string = (
                    f"${line.beta:.3f}^"
                    + "{"
                    + f"{line.sig}"
                    + "}"
                    + f"({df})$"
                )

            lines.append((eff, string))

        return pd.DataFrame(
            lines,
            columns=["\\textbf{Parameters}", "\\textbf{Estimated Value}"],
        )

    @property
    def latex(self):
        header_str = self._latex_header()
        result = self._latex_table()
        latex_list = result.to_latex(
            index=False,
            column_format="ll",
            caption=header_str,
            position="!htbp",
        ).split("\n")

        latex_list.insert(1, "\\centering")
        return latex_list

    @property
    def a_d_value(self) -> dict:
        eff_val = {}
        for e, v in zip(self.effect, self.result["beta"]):
            if e == "m":
                continue
            eff_val[e] = v

        if "d" in eff_val.keys():
            eff_val["d_a"] = eff_val["d"] / eff_val["a"]

        else:
            eff_val["d"] = "/"
            eff_val["d_a"] = "/"

        return eff_val

    @property
    def eff_se(self) -> dict:
        eff_se = {}
        for e, s in zip(self.effect, self.result["se"]):
            if e == "m":
                continue
            eff_se[e] = s

        return eff_se


class VarianceDecomposition(WeightedLeastSquare):
    """
    VarianceDecomposition class inherits from JointScale class.
    It is used for performing variance decomposition analysis.

    Attributes:
        weight (np.ndarray): Diagonal of the precision matrix.

    Methods:
        __init__(self, input_data: str) -> None:
            Initializes the VarianceDecomposition object.
        f_test(f_value, df1, df2) -> str:
            Performs F-test and returns the significance level.
        fit(self, effect) -> None:
            Fits the model and prints the results.
        _print_result(self, sig, ssy, rss, r2, effect) -> None:
            Prints the results in a tabular format.
    """

    def __init__(self, input_data: str) -> None:
        """
        Initializes the VarianceDecomposition object.

        Args:
            input_data (str): Path to the input data file.

        Returns:
            None
        """
        super().__init__(input_data)
        self._weight = np.diag(self._precision_matrix)
        self._raw_result = []
        self.variance = {}

    def fit(self, effect, print_result=True):
        """
        Fits the model and prints the results.

        Args:
            effect (list): List of effects.

        Returns:
            None
        """
        C = np.sum(self._weight * self._y) ** 2 / np.sum(self._weight)
        SSY = float(self._y.T @ self._precision_matrix @ self._y - C)

        _effect = ["m"]
        SSR, MSp, sig_info, rss, r2 = [], [], [], [], []
        effect = self._check_effect(effect, add_m=False)
        for eff in effect:
            _effect.append(eff)
            self._effect = self._check_effect(_effect, add_m=False)
            df = self._y.shape[0] - len(_effect)
            _, beta = self._fit(self._effect)

            _SSR = beta.T @ self._X.T @ self._precision_matrix @ self._y - C
            _SSp = _SSR - (SSR[-1] if SSR else 0)
            _MSp = _SSp

            SSR.append(_SSR)
            MSp.append(_MSp)

            _SSr = SSY - _SSR
            _MSr = _SSr / df
            sig = [(float(_MSp), f_test((_MSp / _MSr), 1, df)) for _MSp in MSp]
            sig_info.append(sig)
            rss.append(float((SSY - np.sum(MSp)) / df))
            r2.append(float(np.sum(MSp) / SSY))

        self._raw_result = [sig_info, SSY, rss, r2, effect]

        for v, e in zip(sig_info[-1], effect):
            self.variance[e] = float(v[0])
        self.variance["SSY"] = SSY
        if print_result:
            self._print_result()

    def _print_result(self):
        """
        Prints the results in a tabular format.

        Args:
            sig (list): List of significance levels.
            ssy (float): Sum of squares for regression.
            rss (list): List of residual sum of squares.
            r2 (list): List of R-squared values.
            effect (list): List of effects.

        Returns:
            None
        """
        sig_infos, ssy, rss, r2, effect = self._raw_result
        num_effect = len(effect)
        lines = [[""] + ["DF", "MS"] * num_effect]

        for idx, eff in enumerate(effect):  # add effect variance line
            _line = [eff]
            for _ in range(idx):
                _line.extend([""] * 2)  # add empty cells

            for sig in sig_infos[idx:]:
                _line.append("1")
                _line.append(f"{sig[idx][0]:.2f}{sig[idx][1]}")
            lines.append(_line)

        _line = ["rss"]
        for idx, r in enumerate(rss):
            _line.append(str(4 - idx))
            _line.append(f"{r:.2f}")
        lines.append(_line)

        lines.append(["SSy"] + ["5", f"{ssy:.2f}"] * num_effect)

        _line = ["r2"]
        for r in r2:
            _line.append("")
            _line.append(f"{r:.4f}")
        lines.append(_line)

        print_list(lines)


def print_list(data):
    max_col_width = []

    for row in data:
        for idx, col in enumerate(row):
            if len(max_col_width) <= idx:
                max_col_width.append(0)
            max_col_width[idx] = max(max_col_width[idx], len(str(col)))

    for row in data:
        for idx, col in enumerate(row):
            print(f"{col:<{max_col_width[idx]}} ", end=" ")
        print()
