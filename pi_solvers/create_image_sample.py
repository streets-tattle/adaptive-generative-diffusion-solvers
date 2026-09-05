import argparse
import os
import pathlib

from pi_solvers.utils import plot_images


def create_image_sample(image_path: str, save_path: str, n: int = 16, n_cols: int = 4):
    if save_path is None:
        save_path = str(pathlib.Path(image_path).parent)

    images = []
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(pathlib.Path(image_path).iterdir()):
        if image.is_file():
            images.append(image)
        if i >= n - 1:
            break
    fig = plot_images(images, n_cols)
    fig.tight_layout(pad=0)
    fig.savefig(save_path + "/samples.png")
    print(f"Image saved to {os.path.abspath(save_path + "/samples.png")}")


def main():
    parser = argparse.ArgumentParser(description="Creates an image in n_cols x n // n_cols. For "
                                                 "good output, ensure n_images is divisible by n_cols")
    parser.add_argument("filepath", type=str,
                        help="Path where the images are stored.")
    parser.add_argument("-o", "--output", default=None, type=str,
                        help="Path where the sample should be written (default current directory).")
    parser.add_argument("-n", "--n_images", default=16, type=int,
                        help="Number of images to use in the sample image (default 16).")
    parser.add_argument("-c", "--n_cols", default=4, type=int,
                        help="Number of columns to use in the sample image grid (default 4).")

    args = parser.parse_args()
    create_image_sample(args.filepath, args.output, args.n_images, args.n_cols)


if __name__ == "__main__":
    main()
