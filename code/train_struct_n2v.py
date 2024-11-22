import logging
import argparse
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.config.transformations import XYFlipModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def run_training(work_dir: str, train_source: str, val_source: str) -> None:
    """
    Runs the CAREamics StructN2V training.

    Parameters
    ----------
    work_dir : str
        Working directory for the experiment.
    train_source : str
        Path to the training data.
    val_source : str
        Path to the validation data.

    Returns
    -------
    None
    """
    print("creating config")
    config = create_n2v_configuration(
        experiment_name="struct_n2v_3D",
        data_type="tiff",
        axes="ZYX",
        patch_size=[64, 128, 128],
        batch_size=4,
        num_epochs=15,
        logger="tensorboard",
        dataloader_params={"num_workers": 15},
        # structN2V parameters
        struct_n2v_axis="horizontal",
        struct_n2v_span=11,
        # disable augmentations because of the noise correlations
        augmentations=[],
    )

    # only use x-axis flip in augmentations
    config.data_config.transforms.insert(
        0,
        XYFlipModel(flip_x=True, flip_y=False),
    )

    careamist = CAREamist(source=config, work_dir=work_dir)
    careamist.train(
        train_source=train_source,
        val_source=val_source,
        use_in_memory=False
    )

def main():
    parser = argparse.ArgumentParser(description='Run CAREamics StructN2V training')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='Working directory for the experiment.')
    parser.add_argument('--train_source', type=str, required=True,
                        help='Path to the training data.')
    parser.add_argument('--val_source', type=str, required=True,
                        help='Path to the validation data.')
    args = parser.parse_args()

    run_training(
        work_dir=args.work_dir,
        train_source=args.train_source,
        val_source=args.val_source
    )

if __name__ == "__main__":
    main()
