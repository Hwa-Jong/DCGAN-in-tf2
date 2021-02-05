import argparse
import os

from training.train_loop import validation

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
        description='DCGAN-Generate',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--load_path', help='model load path', required=True)
    parser.add_argument('--generate_num', help='The number of generated images', default=16, type=int)
    parser.add_argument('--result_dir', help='Root directory for run results', default='results')
    parser.add_argument('--seed', help='Set seed', default=22222, type=int)

    
    args = parser.parse_args()

    validation(**vars(args))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    

# ----------------------------------------------------------------------------

