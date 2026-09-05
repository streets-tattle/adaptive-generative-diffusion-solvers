import pathlib

import scienceplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


if __name__ == "__main__":
    path = "../../data/gaussian_test/grid/h_start_complex"
    var_name = r"$h_0$"

    var = []
    nfe = []
    min_error = []

    for subdir in pathlib.Path(path).iterdir():
        if not subdir.is_file():
            data = pd.read_csv(str(subdir) + "/data.csv")
            var.append(round(float(subdir.name), 4))
            min_error_index = np.argmin(data["pi_error"])
            min_error.append(data["pi_error"][min_error_index])
            nfe.append(data["pi_nfe"][min_error_index])

    plt.figure()
    plt.style.use("science")
    sns.scatterplot(x=var, y=min_error)
    plt.ylabel(r"$D_w$", fontsize=15)
    plt.xlabel(var_name, fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylim(0, 0.25)
    plt.yticks(fontsize=12)
    plt.grid()
    # plt.savefig(path + "/min_error.png")
