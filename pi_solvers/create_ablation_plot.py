import os
from typing import Callable

import torch
import PIL.Image
import tqdm

from pi_solvers import utils
from pi_solvers.solver_lib import PISolver
from pi_solvers.utils import data_logger
from pi_solvers.sde_lib import EDMSDE, SDE
from pi_solvers.solver_lib import *


def full_denoised_at_different_nfe(
        solver_func,
        outdir: str,
        seed: int,
        label: int,
        eval_nfe: tuple[int,...],
        model_url: str,
        device: str = "cuda",
        ode_threshold: float = 0.05
):
    print("Loading model...")
    model, encoder = utils.load_edm_checkpoint(model_url)
    model.to(device)

    sde = EDMSDE().to(device).get_reverse_sde(model, ode_threshold=ode_threshold)

    os.makedirs(outdir, exist_ok=True)

    torch.random.manual_seed(seed)
    noise = torch.randn((1, model.img_channels, model.img_resolution, model.img_resolution), device=device) * 80
    labels = torch.zeros((1, model.label_dim)).to(device)
    labels[:, label] = 1

    for nfe in eval_nfe:
        solver = solver_func(sde, model, nfe).to(device)

        callback = data_logger.LastTimeLogger()
        sde.reset()
        x = solver.solve(noise.clone(), labels=labels, callback=callback)
        t = callback.t
        denoised_x = model(x, t, labels)
        image = encoder.decode(denoised_x).permute(0, 2, 3, 1).cpu().numpy()[0]

        PIL.Image.fromarray(image, "RGB").save(os.path.join(outdir, f"{label}_{nfe}_NFE.png"))


def pi_constructor(**pi_kwargs):
    return lambda sde, _, nfe: construct_heun_end_adaptive_solver(adaptive_solver_class=PISolver,
                                                                  sde=sde,
                                                                  max_iter=nfe // 2,
                                                                  **pi_kwargs)


def ggf_constructor(**ggf_kwargs):
    return lambda sde, _, nfe: construct_heun_end_adaptive_solver(adaptive_solver_class=GottaGoFast,
                                                                  sde=sde,
                                                                  max_iter=nfe // 2,
                                                                  **ggf_kwargs)


def heun_edm_constructor(max_nfe, seed):
    n_steps = (max_nfe + 1) // 2
    discretisation = get_edm_schedule(n_steps)
    def inner(sde, _, nfe):
        n_steps = (nfe + 1) // 2
        steps = discretisation[:n_steps]
        return HeunSolver(sde, steps, seed=seed)
    return inner



if __name__ == "__main__":
    label = 282
    seed = 1

    pi_func = pi_constructor(
        ode_threshold=0.05,
        n_ode_steps=10,
        ki=0.3,
        kp=0.1,
        tau_a=0.06,
        tau_r=18,
        alpha=0.9,
        h_start=40,
        max_decrease=0.2,
        max_increase=5,
        interval=(80, 0.002),
        seed=seed,
        abs_error=False,
        batch_norm=False
    )

    ggf_func = ggf_constructor(
        ode_threshold=0.05,
        n_ode_steps=10,
        tau_a=0.0078,
        tau_r=13.7,
        h_start=25,
        r=0.1,
        alpha=0.7,
        interval=(80, 0.002),
        seed=seed
    )

    heun_func = heun_edm_constructor(49, seed)

    full_denoised_at_different_nfe(
        solver_func=heun_func,
        outdir="../data/ablation_figure/heun_3",
        eval_nfe=(13, 25, 37, 49),
        seed=seed,
        label=label,
        model_url="../model/edm2-img512-l-1879048-0.085.pkl"
    )


