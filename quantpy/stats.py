import numpy as np
import scipy.stats as stats


def t_test(t_value, df) -> str:
    """
    Performs t-test and returns the significance level.

    Args:
        t_value (float): t-value.
        df (int): Degrees of freedom.

    Returns:
        str: Significance level. in terms of '*' or '**'.
    """
    t_thred_05 = stats.t.ppf(0.975, df)
    t_thred_001 = stats.t.ppf(0.995, df)

    t_sig_05 = "*" if np.abs(t_value) > t_thred_05 else ""
    t_sig = "**" if np.abs(t_value) > t_thred_001 else t_sig_05

    if t_sig == "":
        t_sig = "NS"
    return t_sig


def chi2_test(f_value, df) -> str:
    """
    Performs chi-square test and returns the significance level.

    Args:
        f_value (float): F-value.
        df (int): Degrees of freedom.

    Returns:
        str of Significance level. in terms of '*' or '**'.
    """
    chi_thred_05 = stats.chi2.ppf(0.95, df)
    chi_thred_001 = stats.chi2.ppf(0.99, df)
    chi_sig_05 = "*" if f_value > chi_thred_05 else ""
    chi_sig = "**" if f_value > chi_thred_001 else chi_sig_05

    if chi_sig == "":
        chi_sig = "NS"

    return chi_sig


def f_test(f_value, df1, df2) -> str:
    """
    Performs F-test and returns the significance level.

    Args:
        f_value (float): F-value.
        df1 (int): Degrees of freedom for the numerator.
        df2 (int): Degrees of freedom for the denominator.

    Returns:
        str of Significance level. in terms of '*' or '**'.
    """
    f_thred_05 = stats.f.ppf(0.95, df1, df2)
    f_thred_001 = stats.f.ppf(0.99, df1, df2)

    f_sig_05 = "*" if f_value > f_thred_05 else ""
    f_sig = "**" if f_value > f_thred_001 else f_sig_05

    if f_sig == "":
        f_sig = "NS"
    return f_sig
