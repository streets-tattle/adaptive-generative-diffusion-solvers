import argparse
import datetime
import os

from pi_solvers.solver_lib import *
from pi_solvers.evaluation.generate_samples import generate_images
from pi_solvers.utils.data_logger import PIDataLogger
from pi_solvers.utils import write_general_info


def setup_dirs(name: str, path: str = None, exist_okay: bool = False) -> tuple[str, str]:
    if path is None:
        path = f"data/image_testing/{name}/{hash(datetime.time())}"
    image_path = path + "/images/"
    write_path = path + "/data/"

    os.makedirs(image_path, exist_ok=exist_okay)
    os.makedirs(write_path, exist_ok=exist_okay)

    return image_path, write_path


def generate_pi_images(
        batch_size: int,
        device: torch.device,
        n_images: int,
        model: str,
        seed: int,
        output: str,
        ode: bool,
        exist_okay: bool,
        **pi_kwargs
):
    print(f"Setting up PI-solver for {n_images} images...")
    image_path, write_path = setup_dirs("pi_2", output, exist_okay)

    write_general_info(
        path=write_path + "info.txt",
        batch_size=batch_size,
        device=device,
        n_images=n_images,
        model=model,
        seed=seed,
        ode=ode,
        exist_okay=exist_okay,
        **pi_kwargs
    )

    solver_constructor = lambda sde, _: construct_heun_end_adaptive_solver(adaptive_solver_class=PISolver, sde=sde, seed=seed, **pi_kwargs)
    nfe = generate_images(
        solver_func=solver_constructor,
        ode_threshold=0,
        outdir=image_path,
        n_samples=n_images,
        batch_size=batch_size,
        ode=ode,
        seed=seed,
        model_url=model,
        device=device,
        callback=PIDataLogger(write_path=write_path, max_iter=pi_kwargs["max_iter"], batch_size=batch_size)
    )

    with open(write_path + "info.txt", 'a') as f:
        f.write(f"nfe: {nfe}")


def generate_ggf_images(
        batch_size: int,
        device: torch.device,
        n_images: int,
        model: str,
        seed: int,
        output: str,
        ode: bool,
        exist_okay: bool,
        **ggf_kwargs
):
    print(f"Setting up Gotta-Go-Fast-solver for {n_images} images...")
    image_path, write_path = setup_dirs("ggf", output, exist_okay)

    write_general_info(
        path=write_path + "info.txt",
        batch_size=batch_size,
        device=device,
        n_images=n_images,
        model=model,
        seed=seed,
        ode=ode,
        exist_okay=exist_okay,
        **ggf_kwargs
    )

    solver_constructor = lambda sde, _: construct_heun_end_adaptive_solver(adaptive_solver_class=GottaGoFast, sde=sde, seed=seed, **ggf_kwargs)
    nfe = generate_images(
        solver_func=solver_constructor,
        outdir=image_path,
        ode_threshold=0,
        n_samples=n_images,
        batch_size=batch_size,
        ode=ode,
        seed=seed,
        model_url=model,
        device=device,
        callback=PIDataLogger(write_path=write_path, max_iter=ggf_kwargs["max_iter"], batch_size=batch_size)
    )

    with open(write_path + "info.txt", 'a') as f:
        f.write(f"nfe: {nfe}")


def generate_em_images(
        batch_size: int,
        device: torch.device,
        n_images: int,
        model: str,
        seed: int,
        output: str,
        ode: bool,
        exist_okay: bool,
        nfe: int,
        ode_threshold: float,
        rho: float = 7,
        entropy_checkpoint: str = None,
        pi_discretisation: str = None,
        heun: bool = False,
        **kwargs
):
    print(f"Setting up EM-solver for {n_images} images...")
    image_path, write_path = setup_dirs("em", output, exist_okay)

    write_general_info(
        path=write_path + "info.txt",
        batch_size=batch_size,
        device=device,
        n_images=n_images,
        model=model,
        seed=seed,
        ode=ode,
        exist_okay=exist_okay,
        nfe=nfe,
        rho=rho,
        entropy_checkpoint=entropy_checkpoint,
        pi_discretisation=pi_discretisation,
        heun=heun,
        ode_threshold=ode_threshold
    )
    if heun:
        nfe = (nfe + 1) // 2

    if entropy_checkpoint is None and pi_discretisation is None:
        discretisation = get_edm_schedule(nfe, rho=rho)
    elif entropy_checkpoint is not None and pi_discretisation is None:
        discretisation = get_entropy_schedule(nfe, entropy_checkpoint)
    elif entropy_checkpoint is None and pi_discretisation is not None:
        n_steps = int((nfe * 0.8))
        n_ode_steps = nfe - n_steps
        print(n_steps + n_ode_steps)
        discretisation = get_pi_schedule(n_steps, n_ode_steps, pi_discretisation)
    else:
        raise ValueError("Only one of pi_discretisation and entropy_checkpoint should be set.")

    if heun:
        solver_constructor = lambda sde, _: HeunSolver(sde=sde, discretisation=discretisation, seed=seed)
    else:
        solver_constructor = lambda sde, _: EulerMarayumaSolver(sde=sde, discretisation=discretisation, seed=seed)
    generate_images(
        solver_func=solver_constructor,
        outdir=image_path,
        n_samples=n_images,
        batch_size=batch_size,
        ode=ode,
        seed=seed,
        model_url=model,
        device=device,
        ode_threshold=ode_threshold
    )


def generate_edm_images(
        batch_size: int,
        device: torch.device,
        n_images: int,
        model: str,
        seed: int,
        output: str,
        ode: bool,
        exist_okay: bool,
        nfe: int,
        rho: float = 7,
        entropy_checkpoint: str = None,
        pi_discretisation: str = None,
        **edm_kwargs
):
    print(f"Setting up EDM-solver for {n_images} images...")
    image_path, write_path = setup_dirs("edm", output, exist_okay)

    write_general_info(
        path=write_path + "info.txt",
        batch_size=batch_size,
        device=device,
        n_images=n_images,
        model=model,
        seed=seed,
        ode=ode,
        exist_okay=exist_okay,
        nfe=nfe,
        rho=rho,
        entropy_checkpoint=entropy_checkpoint,
        pi_discretisation=pi_discretisation,
        **edm_kwargs
    )
    if entropy_checkpoint is None and pi_discretisation is None:
        discretisation = get_edm_schedule((nfe + 1) // 2, rho=rho)
    elif entropy_checkpoint is not None and pi_discretisation is None:
        discretisation = get_entropy_schedule((nfe + 1) // 2, entropy_checkpoint)
    elif entropy_checkpoint is None and pi_discretisation is not None:
        nfe = nfe + 1
        n_steps = int((nfe * 0.8) // 2)
        n_ode_steps = nfe // 2 - n_steps
        print(n_steps + n_ode_steps)
        discretisation = get_pi_schedule(n_steps, n_ode_steps, pi_discretisation)
    else:
        raise ValueError("Only one of pi_discretisation and entropy_checkpoint should be set.")

    solver_constructor = lambda sde, model: EDMSolver(sde=sde, model=model, discretisation=discretisation, seed=seed, **edm_kwargs)
    generate_images(
        solver_func=solver_constructor,
        outdir=image_path,
        n_samples=n_images,
        batch_size=batch_size,
        ode=ode,
        seed=seed,
        model_url=model,
        device=device
    )


def main():
    # Main argument parsing
    parser = argparse.ArgumentParser(description="Generates images using diffusion using EDM2 models, from sigma=80 to sigma=0",)
    parser.add_argument("-b", "--batch_size", default=48, type=int,
                        help="Batch size for image generation (default 48)")
    parser.add_argument("-d", "--device", default="cuda", type=torch.device,
                        help="Device used for image generation (default cuda)")
    parser.add_argument("-n", "--n_images", default=50000, type=int,
                        help="Number of images to be generated (default 50000)")
    parser.add_argument("--ode", action='store_true',
                        help="Evaluate on ODE instead of SDE")
    parser.add_argument("-m", "--model", default="model/edm2-img64-xl-0671088-0.040.pkl", type=str,
                        help="Model url. Either a locally downloaded checkpoint loadable with dnnlib or an nvidia model url")
    parser.add_argument("-s", "--seed", default=0, type=int,
                        help="Random seed (default 0)")
    parser.add_argument("-o", "--output", default=None, type=str,
                        help="Output directory. If not stated, creates a default directory: data/image_testing/solver_name/")
    parser.add_argument("-e", "--exist_okay", action='store_true',
                        help="Overwrite existing directory if it exists.")
    parser.add_argument("--ode_threshold", default=0.05, type=float,
                        help="Time (noise) threshold from which the solver switches to ODE.")

    subparsers = parser.add_subparsers(title="Solvers")

    # euler-marayuma parser
    em_parser = subparsers.add_parser("euler-marayuma", aliases=["em"],
                                      help="Evaluate images using an Euler-Marayuma solver")
    em_parser.add_argument("--nfe", default=200, type=int,
                           help="Number of function evaluations to use (default 200).")
    em_parser.add_argument("--rho", default=7, type=float,
                           help="Which rho to use for the EDM schedule (default 7)")
    em_parser.add_argument("--entropy_checkpoint", default=None, type=str,
                            help="Which entropy checkpoint to use. Will make EM use an Entropic Time Scheduler: https://arxiv.org/abs/2504.13612")
    em_parser.add_argument("--pi_discretisation", default=None, type=str,
                            help="Which PI paths file to use to compute a discretisation, if provided.")
    em_parser.add_argument("--heun", action="store_true",
                           help="Use Heun sampling instead of EM.")
    em_parser.set_defaults(func=generate_em_images)

    # EDM parser
    edm_parser = subparsers.add_parser("edm",
                                       help="Evaluate images using the EDM solver: https://arxiv.org/pdf/2206.00364")
    edm_parser.add_argument("--nfe", default=64, type=int,
                           help="Number of function evaluations to use (default 64).")
    edm_parser.add_argument("--rho", default=7, type=float,
                           help="Which rho to use for the EDM schedule (default 7)")
    edm_parser.add_argument("--entropy_checkpoint", default=None, type=str,
                            help="Which entropy checkpoint to use. Will make EDM use an Entropic Time Scheduler: https://arxiv.org/abs/2504.13612")
    edm_parser.add_argument("--pi_discretisation", default=None, type=str,
                            help="Which PI paths file to use to compute a discretisation, if provided.")
    edm_parser.add_argument("--S_churn", default=0, type=float,
                            help="Overall amount of stochasticity. If 0, this becomes a probability flow ODE (default 0).")
    edm_parser.add_argument("--S_min", default=0, type=float,
                            help="Minimum time at which stochasticity is added (default 0).")
    edm_parser.add_argument("--S_max", default=float("inf"), type=float,
                            help="Maximum time at which stochasticity is added (default inf).")
    edm_parser.add_argument("--S_noise", default=1, type=float,
                            help="Inflates standard deviation of added noise (default 1).")
    edm_parser.set_defaults(func=generate_edm_images)

    # proportional-integral parser
    pi_parser = subparsers.add_parser("proportional-integral", aliases=["pi"],
                                      help="Evaluate images using our solver.")
    pi_parser.add_argument("--max_iter", default=1000, type=int,
                           help="Maximum number of iterations before terminating (default 1000).")
    pi_parser.add_argument("--ode_threshold", default=0.05, type=float,
                           help="Time (noise) threshold from which the solver switches to discretised Heun on ODE (default 0.2).")
    pi_parser.add_argument("--n_ode_steps", default=3, type=int,
                           help="Number of ODE steps the solver takes after switching to Heun ODE (default 3).")
    pi_parser.add_argument("--ki", default=0.3, type=float,
                           help="Integral constant for step-size control (default 0.3).")
    pi_parser.add_argument("--kp", default=0.1, type=float,
                           help="Proportional constant for step-size control (default 0.1).")
    pi_parser.add_argument("--tau_a", default=0.1, type=float,
                           help="Absolute tolerance (default 0.1).")
    pi_parser.add_argument("--tau_r", default=10, type=float,
                           help="Relative tolerance. Increasing directly decreases NFE (default 10).")
    pi_parser.add_argument("--alpha", default=0.9, type=float,
                           help="Safety factor (default 0.9)")
    pi_parser.add_argument("--h_start", default=30, type=float,
                           help="Starting step size (default 30).")
    pi_parser.add_argument("--max_decrease", default=0.2, type=float,
                           help="Maximum decrease factor in one step of the step size (default 0.05).")
    pi_parser.add_argument("--max_increase", default=5, type=float,
                           help="Maximum increase factor in one step of the step size (default 5).")
    pi_parser.add_argument("--batch_norm", action='store_true',
                           help="Turn on batch normalisation, averaging the discretisation error over each batch, thus using the same step size for each image in the batch.")
    pi_parser.add_argument("--abs_error", action='store_true',
                           help="Turn on absolute error normalisation instead of noise error normalisation.")
    pi_parser.set_defaults(func=generate_pi_images)

    # proportional-integral parser
    ggf_parser = subparsers.add_parser("gotta-go-fast", aliases=["ggf"],
                                      help="Evaluate images using the Gotta Go Fast solver (Jolicoeur-Martineau et al, 2021).")
    ggf_parser.add_argument("--max_iter", default=1000, type=int,
                           help="Maximum number of iterations before terminating (default 1000).")
    ggf_parser.add_argument("--n_ode_steps", default=3, type=int,
                           help="Number of ODE steps the solver takes after switching to Heun ODE (default 3).")
    ggf_parser.add_argument("--tau_a", default=0.0078, type=float,
                           help="Absolute tolerance (default 0.0078).")
    ggf_parser.add_argument("--tau_r", default=10, type=float,
                           help="Relative tolerance. Increasing directly decreases NFE (default 10).")
    ggf_parser.add_argument("--r", default=0.1, type=float,
                            help="Hyperparameter for error scaling (default 0.1).")
    ggf_parser.add_argument("--alpha", default=0.7, type=float,
                           help="Safety factor (default 0.7)")
    ggf_parser.add_argument("--h_start", default=20, type=float,
                           help="Starting step size (default 20).")
    ggf_parser.add_argument("--max_decrease", default=0.2, type=float,
                           help="Maximum decrease factor in one step of the step size (default 0.05).")
    ggf_parser.add_argument("--max_increase", default=5, type=float,
                           help="Maximum increase factor in one step of the step size (default 5).")
    ggf_parser.set_defaults(func=generate_ggf_images)

    args = parser.parse_args()
    args.func(**vars(args))

    print("Finished")


if __name__ == "__main__":
    main()
