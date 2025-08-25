import argparse
from input_conversion import h5_to_tglf_input
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--npy_file")
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()
    f = args.npy_file
    output_dir = args.output_dir
    inputs = np.load(f)
    print(f'Loaded {inputs.shape[0]} inputs at {f}')
    print(inputs)
    # n_ky = 24
    # h5_to_tglf_input.convert_numpy_to_tglf_dirs(inputs, n_ky, out_root=output_dir)
