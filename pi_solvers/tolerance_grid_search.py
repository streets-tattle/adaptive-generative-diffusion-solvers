import argparse
import os

import torch
import pandas as pd

from pi_solvers.evaluation import metrics, hyperparameter_search
from pi_solvers.utils import write_general_info


def main():
    # Main argument parsing
    parser = argparse.ArgumentParser(description="Evaluates the influence of absolute and relative tolerance on NFE.")
    parser.add_argument("tau_a_min", type=float,
                        help="Minimum absolute tolerance")
    parser.add_argument("tau_a_max", type=float,
                        help="Maximum absolute tolerance")
    parser.add_argument("tau_r_min", type=float,
                        help="Minimum relative tolerance")
    parser.add_argument("tau_r_max", type=float,
                        help="Minimum relative tolerance")
    parser.add_argument("resolution", type=int,
                        help="Number of points between min and max tolerance.")
    parser.add_argument("outdir", type=str,
                        help="Directory where data is output.")
    parser.add_argument("-b", "--batch_size", default=64, type=int,
                        help="Batch size for sample generation (default 64)")
    parser.add_argument("-d", "--device", default="cuda", type=torch.device,
                        help="Device used for sample generation (default cuda)")
    parser.add_argument("-e", "--exist_okay", action='store_true',
                        help="Overwrite existing directory if it exists.")
    parser.add_argument("-m", "--model", default="model/edm2-img64-xl-0671088-0.040.pkl", type=str,
                        help="Model url. Either a locally downloaded checkpoint loadable with dnnlib or an nvidia model url")
    parser.add_argument("--ode", action='store_true',
                        help="Evaluate on ODE instead of SDE")
    parser.add_argument("-s", "--seed", default=0, type=int,
                        help="Random seed (default 0)")
    parser.add_argument("--max_iter", default=1000, type=int,
                        help="Maximum number of iterations before terminating (default 1000).")
    parser.add_argument("--ode_threshold", default=0.05, type=float,
                        help="Time (noise) threshold from which the solver switches to discretised Heun on ODE (default 0.2).")
    parser.add_argument("--n_ode_steps", default=5, type=int,
                        help="Number of ODE steps the solver takes after switching to Heun ODE (default 3).")
    parser.add_argument("--ki", default=0.3, type=float,
                        help="Integral constant for step-size control (default 0.3).")
    parser.add_argument("--kp", default=0.1, type=float,
                        help="Proportional constant for step-size control (default 0.1).")
    parser.add_argument("--alpha", default=0.9, type=float,
                        help="Safety factor (default 0.9)")
    parser.add_argument("--h_start", default=0.5, type=float,
                        help="Starting step size (default 0.5).")
    parser.add_argument("--max_decrease", default=0.05, type=float,
                        help="Maximum decrease factor in one step of the step size (default 0.05).")
    parser.add_argument("--max_increase", default=5, type=float,
                        help="Maximum increase factor in one step of the step size (default 5).")
    parser.add_argument("--batch_norm", action='store_true',
                        help="Turn on batch normalisation, averaging the discretisation error over each batch, thus using the same step size for each image in the batch.")
    parser.add_argument("--abs_error", action='store_true',
                        help="Turn on absolute error normalisation instead of noise error normalisation.")
    parser.add_argument("--metric", default=None, type=lambda metric: metrics.Metrics[metric], choices=list(metrics.Metrics),
                        help="Metric to use for grid evaluation. If None, no grid evaluation is executed.")
    parser.add_argument("--ref", default=None, type=str,
                        help="Reference statistics relevant for the metric. If None, no grid evaluation is executed.")

    args = parser.parse_args()

    # Write hyperparameters to file
    print("Writing down hyperparameters...")
    os.makedirs(args.outdir, exist_ok=args.exist_okay)
    write_general_info(args.outdir + "/info.txt", **vars(args))

    # Compute grid
    print("Computing grid...")
    nfes, reject_rate, grid = hyperparameter_search.apply_over_grid(**vars(args))

    # Save grid
    print("Saving grid...")
    df_nfe = pd.DataFrame((nfes).cpu().numpy())
    df_nfe.to_csv(args.outdir + "/nfe.csv")

    df_reject = pd.DataFrame(reject_rate.cpu().numpy())
    df_reject.to_csv(args.outdir + "/reject.csv")

    # Plot grid
    print("Plotting grid...")
    hyperparameter_search.plot_grid(
        grid,
        nfes,
        args.outdir,
        "NFE",
        gamma=0.5
    )
    hyperparameter_search.plot_grid(
        grid,
        reject_rate,
        args.outdir,
        "Reject rate"
    )

    # Calculate metrics
    if args.metric is not None and args.ref is not None:
        print("Calculating metric for all tolerances...")
        rater = hyperparameter_search.MetricRater(
            args.metric,
            torch.load(args.ref),
            args.batch_size,
            args.device
        )
        ratings = hyperparameter_search.evaluate_images(
            args.outdir,
            rater=rater,
            seed=args.seed
        )

        print("Saving ratings...")
        ratings.to_csv(args.outdir + "/ratings.csv")

        # Getting ratings in proper grid form
        ratings = hyperparameter_search.get_ratings_grid(
            grid, ratings
        )

        # Plotting ratings
        print("Generating plots...")
        hyperparameter_search.plot_grid(
            grid,
            ratings,
            args.outdir,
            f"{args.metric.value}",
            gamma=0.5
        )
