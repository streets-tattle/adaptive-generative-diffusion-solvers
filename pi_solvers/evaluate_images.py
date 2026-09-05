import argparse
import os.path

import torch

from pi_solvers.evaluation import feature_vector, metrics


def gen_features(
        image_dir: str,
        batch_size: int = 64,
        device: str = "cuda",
        n_images: int = 0,
        output: str = None,
        statistics_out: str = None,
        **kwargs):
    print("Generating features...")

    if not output:
        output = image_dir + "/data/features.pkl"

    features = feature_vector.detect_image_features(
        image_dir,
        batch_size,
        device=device,
        n_images=n_images,
        save_path=output
    )
    print("Features generated")

    if statistics_out:
        print("Calculating statistics...")
        mean, covariance = metrics.FID.calc_mean_covariance(features)
        print("Saving statistics")
        torch.save({"mu": mean, "sigma": covariance}, statistics_out)


def eval_features(
        sample_dir: str,
        ref_dir: str,
        metric: list[metrics.Metrics],
        sample_output: str = None,
        ref_output: str = None,
        output: str = None,
        ref_statistics: str = None,
        batch_size: int = 64,
        device: str = "cuda",
        n_images: int = 0,
        **kwargs
):
    print("Evaluating images...")
    if os.path.isdir(sample_dir):
        print("Detecting sample features...")
        x_hat = feature_vector.detect_image_features(
            sample_dir,
            batch_size,
            device=device,
            n_images=n_images,
            save_path=sample_output
        )
    else:
        print("Loading sample features...")
        x_hat = torch.load(sample_dir)

    if os.path.isdir(ref_dir):
        print("Detecting reference features...")
        x = feature_vector.detect_image_features(
            ref_dir,
            batch_size,
            detector=feature_vector.InceptionV3Detector(),
            device=device,
            n_images=n_images,
            save_path=ref_output
        )
    else:
        print("Loading reference features...")
        x = torch.load(ref_dir)

    stats = None
    if ref_statistics:
        print("Loading reference statistics...")
        stats = torch.load(ref_statistics, weights_only=False)

    vals = []

    print("Calculating metrics...")
    for m in metric:
        ref = stats if m.uses_stats() else x
        print(f"Calculating {m}")
        data = m.pretty_print(ref, x_hat)
        print(data)
        vals.append(data)
        if output:
            with open(output + "/metrics.txt", "a") as f:
                f.write(data + "\n")

    print("Finished")
    return vals


def main():
    # Main argument parsing
    parser = argparse.ArgumentParser(description="Computes a metric for given images")
    parser.add_argument("-b", "--batch_size", default=64, type=int,
                        help="Batch size for feature generation (default 64)")
    parser.add_argument("-d", "--device", default="cuda", type=torch.device,
                        help="Device used for feature generation (default cuda)")
    parser.add_argument("-n", "--n_images", default=0, type=int,
                        help="Number of images for which features should be generated. 0 means no limit (default 0)")

    subparsers = parser.add_subparsers()

    # gen-features command parsing
    gen_features_parser = subparsers.add_parser("gen-features",
                                                help="Generates feature vectors from images.")
    gen_features_parser.add_argument("image_dir", type=str,
                                     help="Directory in which the images are stored. Images should not be in nested directories.")
    gen_features_parser.add_argument("-o", "--output", default=None, type=str,
                                     help="Optional directory where features are written. By default, writes in the image directory in the new data folder")
    gen_features_parser.add_argument("--statistics_out", default=None, type=str,
                                     help="Optional path where feature statistics should be written. If not given, stats are not computed.")
    gen_features_parser.set_defaults(func=gen_features)

    # eval-features command parsing
    eval_features_parser = subparsers.add_parser("eval-features",
                                                 help="Evaluates image feature vectors")
    eval_features_parser.add_argument("sample_dir", type=str,
                                      help="Directory with the samples (then features are generated) or a saved torch.Tensor whose feature vectors can be used.")
    eval_features_parser.add_argument("ref_dir", type=str,
                                      help="Directory with the reference (then features are generated) or a saved torch.Tensor whose reference feature vectors can be used.")
    eval_features_parser.add_argument("-m", "--metric", action="append", default=[], type=lambda metric: metrics.Metrics[metric], choices=list(metrics.Metrics),
                                      help="Metrics to be computed. Options are FID and MIND.")
    eval_features_parser.add_argument("-s", "--sample_output", default=None, type=str,
                                      help="Optional directory where sample features can be saved if these are computed.")
    eval_features_parser.add_argument("-r", "--ref_output", default=None, type=str,
                                      help="Optional directory where reference features can be saved if these are computed.")
    eval_features_parser.add_argument("-o", "--output", default=None, type=str,
                                      help="Optional file where metric data can be stored.")
    eval_features_parser.add_argument("--ref_statistics", default=None, type=str,
                                      help="Optional path to saved reference statistics")
    eval_features_parser.set_defaults(func=eval_features)

    # Calling function based on which subcommand was passed
    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    # main()
    features = torch.load("../refs/img64_features.pkl")
    mean, covariance = metrics.calc_mean_covariance(features)
    torch.save({"mu": mean, "sigma": covariance}, "../refs/img64_stats.pkl")
