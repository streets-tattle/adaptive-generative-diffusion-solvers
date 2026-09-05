"""Tests various solvers on a 1D Gaussian."""
from typing import Iterable, Callable, Any
import os
import math
import argparse

import scienceplots
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import savgol_filter
import tqdm

from pi_solvers import solver_lib, sde_lib
from pi_solvers.solver_lib import get_pi_schedule
from pi_solvers.utils import gaussians, utils, data_logger

plt.style.use("science")

def calculate_distance(x1: torch.Tensor, x2: torch.Tensor, n_bins=1000) -> float:
    bin_min, bin_max = torch.min(x1[0], x2[0]), torch.max(x1[-1], x2[-1])
    bin_cutoffs = torch.linspace(bin_min, bin_max, n_bins)
    return float(torch.sum(torch.abs(torch.histogram(x1, bins=bin_cutoffs)[0] - torch.histogram(x2, bins=bin_cutoffs)[0])) / x1.shape[0])


def evaluate_solvers(solvers: Iterable[solver_lib.Solver],
                     x_start: torch.Tensor,
                     x: torch.Tensor,
                     sde: sde_lib.SDE,
                     seed: int = 0,
                     n_solvers: int = 100
                     ) -> tuple[list[float], list[float]]:
    nfe, distance = [], []
    for solver in tqdm.tqdm(solvers, total=n_solvers):
        torch.manual_seed(seed)
        sde.reset()
        x_hat = solver.solve(x_start.clone())
        nfe.append(sde.nfe / x.shape[0])
        distance.append(
            scipy.stats.wasserstein_distance(x[:, 0], torch.sort(x_hat)[0][:, 0])
        )
    return nfe, distance


def create_solvers(
        constructor: Callable[[Any], solver_lib.Solver],
        parameter_range: Iterable[Any]
) -> Iterable[solver_lib.Solver]:
    for param in parameter_range:
        yield constructor(param)


def simple_gaussian(sde: sde_lib.LinearDriftSDE):
    n = 10
    mus = lambda i: 2 * i - n
    sigmas = torch.zeros(n) + 0.3
    weights = torch.ones(n) / n

    gaussians_ = []
    for i in range(n):
        gaussians_.append(gaussians.Gaussian(mu=mus(i), sigma=sigmas[i], weight=weights[i]))

    multi_gaussian = gaussians.MultiGaussian(tuple(gaussians_), sde)
    return multi_gaussian


def complex_gaussian(sde: sde_lib.LinearDriftSDE):
    mus = [-12, -10, -8, -5, 0, 4, 9, 13, 15, 17]
    sigmas = [0.5, 1.5, 0.2, 2, 0.3, 1.5, 0.1, 1, 0.2, 0.1]
    weights = torch.tensor([1, 3, 1, 3, 1, 2, 1, 3, 2, 1])
    weights = weights / torch.sum(weights)

    gaussians_ = []
    for i in range(len(mus)):
        gaussians_.append(gaussians.Gaussian(mu=mus[i], sigma=sigmas[i], weight=weights[i]))

    multi_gaussian = gaussians.MultiGaussian(tuple(gaussians_), sde)
    return multi_gaussian


def main():
    # Main argument parsing
    parser = argparse.ArgumentParser(description="Evaluate solvers on 1D gaussians",)
    parser.add_argument("nfe_min", type=int,
                        help="Minimum NFE for the discretised solvers to run at.")
    parser.add_argument("nfe_max", type=int,
                        help="Maximum NFE for the discretised solvers to run at.")
    parser.add_argument("tau_min", type=float,
                        help="Minimum absolute tolerance of the PI solver.")
    parser.add_argument("tau_max", type=float,
                        help="Maximum absolute tolerance of the PI solver.")
    parser.add_argument("-n", "--n_samples", default=100000, type=int,
                        help="Number of samples per solver run (default 100000)")
    parser.add_argument("-r", "--resolution", default=100, type=int,
                        help="Plot resolution, how many points to generate in the NFE/tolerance range.")
    parser.add_argument("--ode", action='store_true',
                        help="Evaluate on ODE instead of SDE")
    parser.add_argument("-g", "--gaussian", default="simple", type=str, choices=["simple", "complex"],
                        help="Which multimodal gaussian to evaluate on.")
    parser.add_argument("-s", "--seed", default=0, type=int,
                        help="Random seed (default 0)")
    parser.add_argument("-o", "--output", default=None, type=str,
                        help="Output directory for data and plots.")
    parser.add_argument("-e", "--exist_okay", action='store_true',
                        help="Overwrite existing directory if it exists.")
    parser.add_argument("--beta_min", default=0.1, type=float,
                        help="Beta min for the VP SDE (default 0.1)")
    parser.add_argument("--beta_max", default=20, type=float,
                        help="Beta max for the VP SDE (default 20)")
    parser.add_argument("--non_adaptive_ref", default=None, type=str,
                        help="File with error and NFE for the Heun and EM solvers.")
    parser.add_argument("--pi_discretisation_tau", default=None, type=float,
                        help="Absolute tolerance at which to generate the PI discretisation")

    # PI Hyperparameters
    parser.add_argument("--max_iter", default=1000, type=int,
                        help="Maximum number of iterations before terminating (default 1000).")
    parser.add_argument("--ki", default=0.3, type=float,
                        help="Integral constant for step-size control (default 0.3).")
    parser.add_argument("--kp", default=0.1, type=float,
                        help="Proportional constant for step-size control (default 0.1).")
    parser.add_argument("--tau_r", default=0, type=float,
                        help="Relative tolerance. Increasing directly decreases NFE (default 0).")
    parser.add_argument("--alpha", default=0.9, type=float,
                        help="Safety factor (default 0.9)")
    parser.add_argument("--h_start", default=0.01, type=float,
                        help="Starting step size (default 30).")
    parser.add_argument("--max_decrease", default=0.2, type=float,
                        help="Maximum decrease factor in one step of the step size (default 0.05).")
    parser.add_argument("--max_increase", default=5, type=float,
                        help="Maximum increase factor in one step of the step size (default 5).")
    parser.add_argument("--batch_norm", action='store_true',
                        help="Turn on batch normalisation, averaging the discretisation error over each batch, thus using the same step size for each image in the batch.")
    parser.add_argument("--abs_error", action='store_true',
                        help="Turn on absolute error normalisation instead of noise error normalisation.")


    args = parser.parse_args()

    # Write hyperparameters to file
    print("Writing down hyperparameters...")
    os.makedirs(args.output, exist_ok=args.exist_okay)
    utils.write_general_info(args.output + "/info.txt", **vars(args))

    # Set seed
    torch.manual_seed(args.seed)

    # Create the SDE
    sde = sde_lib.LinearVariancePreservingSDE(args.beta_min, args.beta_max)

    # Get score function
    if args.gaussian == "simple":
        multi_gaussian = simple_gaussian(sde)
    else:
        multi_gaussian = complex_gaussian(sde)

    score_func = multi_gaussian.get_score_function()

    # Create reverse sde based on score function
    reverse_sde = sde.get_reverse_sde(score_func)

    # Create data
    samples = multi_gaussian.sample(args.n_samples).unsqueeze(-1)
    x = torch.sort(samples)[0]
    x_start = sde.sample(x, torch.Tensor([1]))

    # Create solver iterables
    n_evaluation_points = args.resolution

    em_evaluation_range = torch.linspace(args.nfe_min, args.nfe_max, n_evaluation_points)
    em_constructor = lambda n_steps: solver_lib.EulerMarayumaSolver(reverse_sde, torch.linspace(1, 0, int(n_steps)), seed=args.seed)
    heun_constructor = lambda n_steps: solver_lib.HeunSolver(reverse_sde, torch.linspace(1, 0, int(n_steps) // 2))

    pi_evaluation_range = torch.exp(
        torch.linspace(math.log(args.tau_min), math.log(args.tau_max), n_evaluation_points))

    pi_constructor = lambda tolerance: solver_lib.PISolver(
        reverse_sde,
        ki=args.ki,
        kp=args.kp,
        tau_a=tolerance,
        tau_r=args.tau_r,
        alpha=args.alpha,
        h_start=args.h_start,
        max_decrease=args.max_decrease,
        max_increase=args.max_increase,
        batch_norm=args.batch_norm,
        abs_error=args.abs_error,
        max_iter=args.max_iter,
        seed=args.seed
    )

    em_solvers = create_solvers(em_constructor, em_evaluation_range)
    heun_solvers = create_solvers(heun_constructor, em_evaluation_range)
    pi_solvers = create_solvers(pi_constructor, pi_evaluation_range)

    # Evaluate
    if args.non_adaptive_ref is None:
        df = pd.DataFrame()
        print("Evaluating Euler-Marayuma")
        df["em_nfe"], df["em_error"] = evaluate_solvers(em_solvers, x_start, x, reverse_sde, args.seed, n_solvers=args.resolution)

        print("Evaluating Heun")
        df["heun_nfe"], df["heun_error"] = evaluate_solvers(heun_solvers, x_start, x, reverse_sde, args.seed, n_solvers=args.resolution)
    else:
        df = pd.read_csv(args.non_adaptive_ref)

    print("Evaluating PI")
    # df["pi_nfe"], df["pi_error"] = evaluate_solvers(pi_solvers, x_start, x, reverse_sde, args.seed, n_solvers=args.resolution)
    # df["pi_tau"] = pi_evaluation_range

    # print("Evaluating Heun with PI discretisation")
    # if args.pi_discretisation_tau is not None:
    #     solver = pi_constructor(args.pi_discretisation_tau)
    #     callback = data_logger.PIDataLogger(args.output + "/", batch_size=args.n_samples, max_iter=args.max_iter)
    #     solver.solve(x_start, callback=callback)
    #     callback.write()

    #     heun_pi_constructor = lambda n_steps: solver_lib.HeunSolver(reverse_sde, get_pi_schedule(n_steps=int(n_steps) // 2, n_ode_steps=0, pi_paths_file=args.output + "/_t.csv", t_max=1, t_min=0, t_ode=0))
    #     heun_pi_solvers = create_solvers(heun_pi_constructor, em_evaluation_range)
    #     df["heun_pi_nfe"], df["heun_pi_error"] = evaluate_solvers(heun_pi_solvers, x_start, x, reverse_sde, args.seed, n_solvers=args.resolution)

    # Write data
    print("Saving data, creating plots")
    df.to_csv(args.output + "/data.csv")

    # Smooth distances
    em_error_smooth = savgol_filter(df["em_error"], window_length=9, polyorder=3)
    heun_error_smooth = savgol_filter(df["heun_error"], window_length=9, polyorder=3)
    pi_error_smooth = savgol_filter(df["pi_error"], window_length=9, polyorder=3)
    if args.pi_discretisation_tau is not None:
        heun_pi_error_smooth = savgol_filter(df["heun_pi_error"], window_length=9, polyorder=3)

    # Create plot
    fig = plt.figure(dpi=300)
    fig.set_size_inches(5, 3.75)
    plt.plot(df["em_nfe"], em_error_smooth, label="Euler-Maruyama")
    plt.plot(df["heun_nfe"], heun_error_smooth, label="Stochastic Heun")
    if args.pi_discretisation_tau is not None:
        plt.plot(df["heun_pi_nfe"], heun_pi_error_smooth, label="PI Static Stochastic Heun")
    plt.plot(df["pi_nfe"], pi_error_smooth, label="Proportional-integral")
    plt.xlabel("NFE", fontsize=15)
    plt.xlim(0, 200)
    plt.xticks(fontsize=12)
    plt.yscale("log")
    plt.ylabel(r"$D_W$", fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(args.output + "/plot.png")


if __name__ == "__main__":
    main()
