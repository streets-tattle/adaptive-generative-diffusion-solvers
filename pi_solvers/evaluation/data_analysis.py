import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import torch

from BitstreamDiffusion.diffusion.continuous.samplers import SigmaSchedule
from BitstreamDiffusion.diffusion.continuous.processes import ContinuousForwardProcess
from BitstreamDiffusion.evaluation.utils import load_config

from pi_solvers.solver_lib import get_edm_schedule
from pi_solvers.utils import compute_discretisation_interpolation


plt.style.use("science")


def detect_outliers(paths: np.ndarray, t_min: float = 0.05, n_outlier_steps: int = 15):
    lengths = []

    for i, path in enumerate(paths):
        end_index = np.where(path <= t_min)[0][0]
        path = path[:(end_index + 1)]

        if end_index <= 2:
            continue

        if path.shape[0] < n_outlier_steps:
            print(f"Outlier at {i} with lengths {path.shape[0]}")

        lengths.append(path.shape[0])

    plt.figure(figsize=(5, 3.75))
    plt.hist(lengths)
    plt.ylabel("Count")
    plt.xlabel("Number of steps taken")
    plt.show()



def get_paths(data_path: str, t_max: float, t_min: float, plot_res: int):
    # Load in t csv
    ts = pd.read_csv(data_path + "/_t.csv").to_numpy()
    # Add start time to histogram
    ts[:, 0] = np.full(ts.shape[0], t_max)

    t_grid, paths = compute_discretisation_interpolation(ts, plot_res, t_min=t_min)

    means, stds = paths.mean(axis=0), paths.std(axis=0)

    t_grid = np.linspace(0, 1, means.shape[0])

    return t_grid, means, stds, paths


def generate_pi_image_trajectories(ax, label: str, data_path: str, t_max: float = 80, t_min: float = 0.002, t_ode:float = 0.05, plot_res: int = 200, color: str = "r", n_paths: int = 3, seed=0):
    np.random.seed(seed)

    if t_ode != t_min:
        discretisation = get_edm_schedule(int(plot_res * 0.2), t_min=t_min, t_max=t_ode)[:-1]
        plot_res = int(plot_res * 0.8)

    t_grid, means, stds, paths = get_paths(data_path, t_max, t_ode, plot_res)

    random_paths = paths[np.random.randint(0, paths.shape[0], n_paths), :]

    print("Plotting...")
    if t_ode != t_min:
        means = np.concat([means, discretisation])
        t_grid = np.linspace(0, 1, means.shape[0])
        ax.plot(t_grid, means, label=label, c=color)
        ax.fill_between(t_grid[:plot_res], means[:plot_res] + stds, means[:plot_res] - stds, alpha=0.1, color=color)
    else:
        ax.plot(t_grid, means, label=label, c=color)
        ax.fill_between(t_grid, means + stds, means - stds, alpha=0.1, color=color)

    # Plot random paths
    for i in range(n_paths):
        ax.plot(t_grid, random_paths[i, :], label=f"Sample Path {i}", linewidth=1)

    # plt.plot(np.linspace(0, 1, discretisation.shape[0]), discretisation, label="EDM Discretisation")
    ax.legend()
    plt.yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Fraction along SDE path $i/N$")
    ax.set_ylim(t_min, t_max)
    ax.set_ylabel(r"$\sigma$")
    ax.grid()
    return ax


def analyse_pi_gaussian_trajectories(ax, label: str, data_path: str, t_max: float = 1, t_min: float = 0, plot_res: int = 200, color: str = "r", n_paths: int = 3, seed=0):
    np.random.seed(seed)
    t_grid, means, stds, paths = get_paths(data_path, t_max, t_min, plot_res)

    random_paths = paths[np.random.choice(paths.shape[0], n_paths), :]

    print("Plotting...")
    # Plotting
    ax.plot(t_grid, means, label=label, c=color)
    ax.fill_between(t_grid, means - stds, means + stds, alpha=0.1, color=color)

    # Plot random paths
    for i in range(n_paths):
        ax.plot(t_grid, random_paths[i, :], label=f"Sample Path {i}", linewidth=1)

    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Fraction along SDE path $i/N$")
    ax.set_ylim(t_min, t_max)
    ax.set_ylabel(r"$t$")
    ax.grid()
    return ax


if __name__ == "__main__":
    data_path = "../../data/image_testing/pi_2/75NFE_2/data"
    t_min = 0.002
    t_max = 80

    fig = plt.figure(dpi=300)
    # creating a dictionary
    font = {'size': 12, 'weight': "bold"}

    # using rc function
    plt.rc('font', **font)
    fig.set_size_inches(5, 3.75)
    ax = fig.add_subplot(111)
    discretisation = get_edm_schedule(200, t_min=t_min)[:-1]
    ax.plot(np.linspace(0, 1, discretisation.shape[0]), discretisation, label="EDM Schedule", c="y")
    cfg = load_config("../../BitstreamDiffusion/configs/lm1b/continuous/eval/rate_eval_seeds.py")

    sigmas = SigmaSchedule(ContinuousForwardProcess(cfg), cfg, torch.device("cpu"))
    sigmas_ = sigmas.prepare(
        schedule="entropic",
        num_steps=200,
        entropy_run_dir="../../BitstreamDiffusion/assets/entropy_tables/lm1b",
    )

    linear = torch.linspace(0, 1, sigmas_.shape[0])
    ax.plot(linear, sigmas_, label="Entropy LM1B")
    generate_pi_image_trajectories(ax, data_path=data_path  , label="PI Static ImageNet-64 (ours)", t_min=t_min, t_max=t_max, n_paths=0, color="r")
    generate_pi_image_trajectories(ax, data_path="../../data/image_testing/pi/Cluster/ffhq_test/data", label="PI Static FFHQ (ours)", t_min=t_min, t_max=t_max, n_paths=0, color="g")
    generate_pi_image_trajectories(ax, data_path="../../data/text_data/pi_files/run2", label="PI Static LM1B (ours)", t_ode=t_min, t_min=t_min, t_max=t_max, n_paths=0, color="b")
    # analyse_pi_gaussian_trajectories(ax, data_path=data_path, label="PI Average", t_max=1, t_min=0, n_paths=3)



    fig.show()


    ts = pd.read_csv(data_path + "/_t.csv").to_numpy()
    # Add start time to histogram
    ts[:, 0] = np.full(ts.shape[0], t_max)

    detect_outliers(ts, t_min)