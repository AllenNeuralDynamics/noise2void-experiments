import os
import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage.exposure import rescale_intensity

from careamics import CAREamist


def run_prediction(model_path: str, data_dir: str, outdir: str) -> None:
    """
    Runs predictions using a trained CAREamics model.

    Parameters
    ----------
    model_path : str
        Path to the trained model checkpoint.
    data_dir : str
        Directory containing input data (TIFF files) for prediction.
    outdir : str
        Output directory where predictions will be saved.

    Returns
    -------
    None
    """
    # Create CAREamics model from checkpoint
    careamist = CAREamist(model_path)

    os.makedirs(outdir, exist_ok=True)

    for filename in os.listdir(data_dir):
        tiff_path = os.path.join(data_dir, filename)
        image = tifffile.imread(tiff_path)
        mn, mx = image.min(), image.max()

        pred = careamist.predict(
            source=tiff_path,
            tile_size=[128, 128, 128],
            tile_overlap=[16, 16, 16],
            tta=True
        )[0].squeeze()

        pred = rescale_intensity(pred, in_range="image", out_range=(mn, mx)).astype(np.uint16)

        print(f"Processed {filename}, output shape: {pred.shape}")
        output_path = os.path.join(outdir, Path(tiff_path).name)
        tifffile.imwrite(output_path, pred)


def main():
    parser = argparse.ArgumentParser(description='Run predictions using a trained CAREamics model.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing input data (TIFF files) for prediction.')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory where predictions will be saved.')
    args = parser.parse_args()

    run_prediction(
        model_path=args.model_path,
        data_dir=args.data_dir,
        outdir=args.outdir
    )

if __name__ == "__main__":
    main()
