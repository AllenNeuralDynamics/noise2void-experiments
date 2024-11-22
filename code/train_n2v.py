import logging
import argparse
from careamics import CAREamist
from careamics.config import create_n2v_configuration

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def run_training(work_dir: str, train_source: str, val_source: str, use_n2v2: bool) -> None:
    """
    Runs the CAREamics N2V or N2V2 training.

    Parameters
    ----------
    work_dir : str
        Working directory for the experiment.
    train_source : str
        Path to the training data.
    val_source : str
        Path to the validation data.
    use_n2v2 : bool
        Flag to use N2V2 algorithm if True, N2V if False.

    Returns
    -------
    None
    """
    print("creating config")
    experiment_name = "n2v2_3D" if use_n2v2 else "n2v_3D"
    config = create_n2v_configuration(
        experiment_name=experiment_name,
        data_type="tiff",
        axes="ZYX",
        patch_size=[64, 128, 128],
        batch_size=4,
        num_epochs=15,
        logger="tensorboard",
        dataloader_params={"num_workers": 15},
        use_n2v2=use_n2v2
    )

    careamist = CAREamist(source=config, work_dir=work_dir)
    careamist.train(
        train_source=train_source,
        val_source=val_source,
        use_in_memory=False
    )

def main():
    parser = argparse.ArgumentParser(description='Run CAREamics N2V/N2V2 training')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='Working directory for the experiment.')
    parser.add_argument('--train_source', type=str, required=True,
                        help='Path to the training data.')
    parser.add_argument('--val_source', type=str, required=True,
                        help='Path to the validation data.')
    parser.add_argument('--use_n2v2', action='store_true',
                        help='Use N2V2 algorithm if set, else use N2V.')
    args = parser.parse_args()

    run_training(
        work_dir=args.work_dir,
        train_source=args.train_source,
        val_source=args.val_source,
        use_n2v2=args.use_n2v2
    )

if __name__ == "__main__":
    main()
