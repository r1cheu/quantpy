from quantpy import JointScale, VarianceDecomposition

if __name__ == "__main__":
    mean_var = "./workdir/data/C1_HD_mean_var.tsv"
    pheno = "HD"
    pop = "C1"
    js = JointScale(mean_var, pheno, pop)

    js.fit(["a", "d"])

    vd = VarianceDecomposition(mean_var)
    vd.fit(["a", "d", "ad", "aa"])
