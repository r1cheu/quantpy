import numpy as np


def eq_genotype_eff(genotypes: np.ndarray, y: np.ndarray):
    def _cal_eff(genotype: int):
        return (
            np.sum(y[genotypes == genotype]) / np.bincount(genotypes)[genotype]
        )

    return np.array([_cal_eff(i) for i in range(3)])


class EqPopMeanEffect:
    def __init__(self, f: np.ndarray, y: np.ndarray):
        self.pA, self.pa = f
        self.AA, self.Aa, self.aa = y
        self._compute_genotype_scale()

    def _compute_genotype_scale(self):
        self.m = (self.AA + self.aa) / 2
        self.a = self.AA - self.m
        self.d = self.Aa - self.m

    @property
    def M(self):
        return self.a * (self.pA - self.pa) + 2 * self.d * self.pA * self.pa

    @property
    def M_prime(self):
        return self.M + self.m

    @property
    def inbreeding_single_loci(self):
        raise NotImplementedError

    @property
    def A_effect(self):
        return self.pA * self.a + self.pa * self.d - self.M

    @property
    def a_effect(self):
        return self.pA * self.d - self.pa * self.a - self.M

    @property
    def AA_effect(self):
        return self.a - self.M

    @property
    def Aa_effect(self):
        return self.d - self.M

    @property
    def aa_effect(self):
        return -self.a - self.M

    @property
    def A_sub_a_effect(self):
        return self.A_effect - self.a_effect

    @property
    def a_sub_A_effect(self):
        return -self.A_sub_a_effect

    @property
    def AA_breed_value(self):
        return 2 * self.A_effect

    @property
    def Aa_breed_value(self):
        return self.A_effect + self.a_effect

    @property
    def aa_breed_value(self):
        return 2 * self.a_effect

    @property
    def AA_dom_bias(self):
        return self.AA_effect - self.AA_breed_value

    @property
    def Aa_dom_bias(self):
        return self.Aa_effect - self.Aa_breed_value

    @property
    def aa_dom_bias(self):
        return self.aa_effect - self.aa_breed_value

    @property
    def d_by_a(self):
        return (
            self.d / np.abs(self.a)
        )  # a should be positive since we not assign the A or a (A_1, A_2 here)
