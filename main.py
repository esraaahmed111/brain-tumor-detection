import argparse
from classifier import build_and_train
from segmentation import run_segmentation_training


# Configuration
def get_args():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection and Segmentation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classify", "segment", "all"],
        default="all",
        help="classify | segment | all"
    )
    return parser.parse_args()


def main():
    args = get_args()

    if args.mode == "classify":
        print("Running Classification Training...")
        build_and_train()

    elif args.mode == "segment":
        print("Running Segmentation Training...")
        run_segmentation_training()

    elif args.mode == "all":
        print("Running Classification Training...")
        build_and_train()
        print("Running Segmentation Training...")
        run_segmentation_training()


if __name__ == "__main__":
    main()
