import argparse
import os

from training.train_loop import train

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# ----------------------------------------------------------------------------
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='DCGAN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset_dir', help='Training dataset path', required=True, type=str)
    parser.add_argument('--result_dir', help='Root directory for run results', default='results', type=str)
    parser.add_argument('--load_path', help='model load path', default=None)
    parser.add_argument('--batch_size', help='Batch size', default=128, type=int)
    parser.add_argument('--epochs', help='Epochs', default=100, type=int)
    parser.add_argument('--lr', help='Learning-rate', default=0.0002, type=float)
    parser.add_argument('--save_term', help='Model save term', default=10)
    
    args = parser.parse_args()

    train(**vars(args))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    

# ----------------------------------------------------------------------------

