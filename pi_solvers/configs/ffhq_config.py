from ml_collections import config_dict

from pi_solvers.evaluation.metrics import Metrics


def get_config():
    cfg = config_dict.ConfigDict()

    # Sampling
    cfg.model = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl"
    cfg.n_samples = 50000
    cfg.sampling_batch_size = 128
    cfg.device = "cuda:0"
    cfg.seed = 0
    cfg.ode = False
    cfg.exist_okay = True
    cfg.base_out = "data/ffhq/"

    # Evaluation
    cfg.metrics = [Metrics.FID]
    cfg.feature_path = "refs/ffhq_features.pkl"
    cfg.stats_path = "refs/ffhq_stats.pkl"
    cfg.eval_batch_size = 1024
    cfg.metrics_out = cfg.base_out

    cfg.specs = []

    def add_spec(
            sampler_name: str,
            sampler_schedule_name: str,
            nfe: int,
            ode_threshold: float = 0.05,
            sampler_schedule_path: str | None = None,
            **sampler_kwargs
    ):
        spec = config_dict.ConfigDict()
        spec.sampler_name = sampler_name
        spec.sampler_schedule_name = sampler_schedule_name
        spec.nfe = nfe
        spec.sampler_schedule_path = sampler_schedule_path
        spec.sampler_kwargs = sampler_kwargs
        spec.out_path = f"{cfg.base_out}/{sampler_name}_{sampler_schedule_name}/{nfe}NFE/"
        spec.ode_threshold = ode_threshold
        cfg.specs.append(spec)

    # PI specs
    pi_params = lambda tau_rel, h_0, n: {
        "n_ode_steps": n,
        "ki": 0.3,
        "kp": 0.1,
        "tau_a": 0.06,
        "tau_r": tau_rel,
        "alpha": 0.9,
        "h_start": h_0,
        "max_decrease": 0.2,
        "max_increase": 5,
        "max_iter": 1000
    }

    add_spec("pi", "adaptive", 49, **pi_params(19.2, 40, 5))
    add_spec("pi", "adaptive", 75, **pi_params(15.3, 35, 7))
    add_spec("pi", "adaptive", 99, **pi_params(12, 35, 10))

    # GGF specs
    ggf_params = lambda tau_rel, h_0, n: {
        "ode_threshold": 0.05,
        "n_ode_steps": n,
        "tau_a": 0.0078,
        "tau_r": tau_rel,
        "alpha": 0.7,
        "h_start": h_0,
        "max_decrease": 0.2,
        "max_increase": 5,
        "r": 0.1,
        "max_iter": 1000
    }
    add_spec("ggf", "adaptive", 49, **ggf_params(25.4, 25, 5))
    add_spec("ggf", "adaptive", 75, **ggf_params(17.15, 25, 7))
    add_spec("ggf", "adaptive", 99, **ggf_params(13.7, 20, 10))

    # EM
    add_spec("euler-maruyama", "edm", 49)
    add_spec("euler-maruyama", "edm", 75)
    add_spec("euler-maruyama", "edm", 99)

    # Stochastic Heun
    add_spec("heun", "edm", 49)
    add_spec("heun", "edm", 75)
    add_spec("heun", "edm", 99)

    add_spec("heun", "pi", 49, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/49NFE/data/_t.csv")
    add_spec("heun", "pi", 75, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/75NFE/data/_t.csv")
    add_spec("heun", "pi", 99, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/99NFE/data/_t.csv")

    # EDM
    edm_churn_params = {
        "S_churn": 40,
        "S_min": 0.05,
        "S_max": 50,
        "S_noise": 1.003
    }

    add_spec("edm", "edm", 49, **edm_churn_params)
    add_spec("edm", "edm", 75, **edm_churn_params)
    add_spec("edm", "edm", 99, **edm_churn_params)

    add_spec("edm", "pi", 49, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/49NFE/data/_t.csv", **edm_churn_params)
    add_spec("edm", "pi", 75, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/75NFE/data/_t.csv", **edm_churn_params)
    add_spec("edm", "pi", 99, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/99NFE/data/_t.csv", **edm_churn_params)

    return cfg
