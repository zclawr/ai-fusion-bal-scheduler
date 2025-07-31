import os
import subprocess
import numpy as np

def run_pyro_convert(input_path: str, output_dir: str, output_filename: str) -> None:
    """
    Run `pyro convert CGYRO inputfile -o output_path` from Python.

    Args:
        input_path (str): Path to the CGYRO input file.
        output_dir (str): Directory where the output file should be saved.
        output_filename (str): Output file name (e.g., 'converted.h5').

    Raises:
        RuntimeError: If the conversion command fails.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    command = ["pyro", "convert", "CGYRO", input_path, "-o", output_path]
    print(f"Running command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{result.stderr}")
    else:
        print(f"Conversion successful. Output saved to {output_path}")    

if __name__ == "__main__":
    run_pyro_convert(
        input_path="/global/homes/w/wyl002/Github/ai-fusion-gknn-wesley/tglf_outputs/tglf_input_20250719_135322/sample_0/ky_0/input.tglf",
        output_dir="./converted_outputs",
        output_filename="my_output.cgyro"
    )
