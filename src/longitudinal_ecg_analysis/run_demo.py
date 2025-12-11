# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
run_demo.py

Runs the key scripts in the longitudinal_ecg_analysis package
Usage:
python -m longitudinal_ecg_analysis.run_demo <dataset_root_raw_folder> <dataset_root_proc_folder> <dataset_name>
"""

import sys
import subprocess
from pathlib import Path
from longitudinal_ecg_analysis.utils import function_start_end_print

if __name__ == "__main__":
    import sys

    # Check whether expected number of inputs have been provided
    if len(sys.argv) != 4:
        print("Usage: python -m longitudinal_ecg_analysis.run_demo <dataset_root_raw_folder> <dataset_root_proc_folder> <dataset_name>")
        sys.exit(1)

    # Parse inputs
    dataset_root_raw_folder = sys.argv[1]
    dataset_root_proc_folder = sys.argv[2]
    dataset_name = sys.argv[3]

    # Specify analysis name
    analysis_name = 'demo_analysis'

    # Start-matter
    function_start_end_print()
    print(f"Running demo analysis of '{dataset_name}' dataset.")

    # Run each module

    subprocess.run([
        sys.executable, "-m", "longitudinal_ecg_analysis.gen_dataset_settings",
        dataset_root_raw_folder, dataset_root_proc_folder, dataset_name
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "longitudinal_ecg_analysis.curate_entire_dataset",
        dataset_root_proc_folder
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "longitudinal_ecg_analysis.gen_analysis_settings",
        dataset_root_proc_folder, analysis_name, "True"
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "longitudinal_ecg_analysis.curate_analysis_dataset",
        dataset_root_proc_folder, analysis_name
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "longitudinal_ecg_analysis.derive_signal_features",
        dataset_root_proc_folder, analysis_name
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "longitudinal_ecg_analysis.compile_for_stats",
        dataset_root_proc_folder, analysis_name
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "longitudinal_ecg_analysis.stats_analysis",
        dataset_root_proc_folder, analysis_name
    ], check=True)

    # End-matter
    print("\nDemo pipeline complete.")
    function_start_end_print()
