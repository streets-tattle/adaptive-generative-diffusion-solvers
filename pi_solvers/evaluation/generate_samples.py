import os
from typing import Callable

import torch
import PIL.Image
import tqdm

from pi_solvers import utils
from pi_solvers.utils.data_logger import DataLogger
from pi_solvers.sde_lib import EDMSDE, SDE
from pi_solvers.solver_lib.solvers import Solver


def generate_images(
        solver_func: Callable[[SDE, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]], Solver],
        outdir: str,
        seed: int = 0,
        n_samples: int = 50000,
        batch_size: int = 64,
        ode: bool = True,
        ode_threshold: float = 0,
        model_url: str = "../model/edm2-img64-xl-0671088-0.040.pkl",
        device: torch.device | str = "cuda",
        callback: DataLogger | None = None
):
    os.makedirs(outdir, exist_ok=True)
    seeds = range(seed, n_samples + seed)

    print("Loading model...")
    model, encoder = utils.load_edm_checkpoint(model_url)
    model.to(device)

    sde = EDMSDE(ode=ode).to(device).get_reverse_sde(model, ode_threshold=ode_threshold)
    solver = solver_func(sde, model).to(device)

    nfe = 0

    print("Sampling images...")
    # Sampling loop
    for i in tqdm.tqdm(range(0, n_samples, batch_size)):

        # Bound batch size if new batch would exceed total amount of samples
        if (n_samples - i) < batch_size:
            batch_size = n_samples - i

        # Get seeds for the batch
        batch_seeds = seeds[i:(i + batch_size)]

        # Get noise and labels
        rng = utils.StackedRandomGenerator(device, batch_seeds)
        noise = rng.randn((batch_size, model.img_channels, model.img_resolution, model.img_resolution), device=device) * 80
        if model.label_dim > 0:
            labels = torch.eye(model.label_dim, device=device)[
                rng.randint(model.label_dim, size=[len(batch_seeds)], device=device)]
        else:
            labels = None

        # Sample using generated noise
        images = solver.solve(noise, labels, callback)

        # Save images
        if labels is None:
            labels = torch.zeros(batch_size)
        for seed, image, label in zip(batch_seeds, encoder.decode(images).permute(0, 2, 3, 1).cpu().numpy(), labels):
            if labels is not None:
                label = torch.argmax(label)
            PIL.Image.fromarray(image, "RGB").save(os.path.join(outdir, f"{seed:06d}-{label}.png"))

        if callback is not None:
            callback.write()

        nfe += sde.nfe
        if i == 0:
            print(f"First iteration NFE: {nfe / batch_size}")
        sde.reset()

    return nfe / n_samples