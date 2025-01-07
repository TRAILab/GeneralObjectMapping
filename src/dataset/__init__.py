from src.dataset.scannet import ScanNet


def init_dataset(args):
    """
    Load dataset from: ScanNet, CO3D
    """
    if args.dataset_name == "scannet":
        dataset = ScanNet(args.sequence_dir)
    elif args.dataset_name == "co3d":
        from src.dataset.co3d.co3d import CO3D

        dataset = CO3D(args.sequence_dir, args.dataset_category, args.co3d_subset_name)

    return dataset
