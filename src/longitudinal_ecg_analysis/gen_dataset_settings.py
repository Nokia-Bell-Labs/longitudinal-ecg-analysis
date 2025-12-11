# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
gen_dataset_settings.py

Generates a JSON settings file for a given dataset. Can be run as a script
or imported as a module. The settings file includes paths to input data,
output locations, and analysis parameters.
"""
import sys
import json
from pathlib import Path
from longitudinal_ecg_analysis.utils import standardise_name, ensure_directories_exist, obtain_dataset_settings_file_path, function_start_end_print

def _generate_dataset_settings_file(
    dataset_root_raw_folder: str,
    dataset_root_proc_folder: str,
    dataset_name: str,
):
    """Generate a JSON settings file for the dataset.

    Args:
        dataset_root_raw_folder (str): Path to folder containing raw input data.
        dataset_root_proc_folder (str): Path to folder in which to save settings.
        dataset_name (str): Name of the dataset (e.g. music).

    Returns:
        None. Writes the settings file to disk.
    """
    
    # Start-matter
    function_start_end_print()
    print(f"Generating settings file for the '{dataset_name}' dataset.")
    
    # setup paths
    dataset_root_raw_folder = Path(dataset_root_raw_folder).resolve()
    dataset_root_proc_folder = Path(dataset_root_proc_folder).resolve()
    settings_file_path = obtain_dataset_settings_file_path(dataset_root_proc_folder)  # path at which to save dataset settings file

    # specify dataset settings
    settings = {
        "dataset_name": dataset_name,
        "paths": {
            "dataset_root_raw_folder": str(dataset_root_raw_folder),
            "dataset_root_proc_folder": str(dataset_root_proc_folder),
        },
        "terms": {
            "outcome_prefix": 'response',
            "clinical_prefix": 'clinical',
        },
        "parameters": {
            "alpha": 0.05,
            "signal_duration_to_analyse": 20*60,
            "window_duration": 10*60,
            "epoch_duration": 5*60,
            "buffer_to_discard": 60,
            "max_no_subjs": 20,
        },
        "redo_curation": False,
        "redo_derive_features": False,
    }

    # Additional processed data paths
    # - folders
    settings["paths"]["entire_dataset_proc_folder"] = str(Path(settings["paths"]["dataset_root_proc_folder"]) / 'entire_dataset')
    settings["paths"]["derived_features_proc_folder"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'derived_features')
    settings["paths"]["preprocessing_proc_folder"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'preprocessing')
    # - initial variables
    settings["paths"]["variables_subject_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'variables_subject_all.csv')
    settings["paths"]["variables_encounter_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'variables_encounter_all.csv')
    settings["paths"]["variables_segment_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'variables_segment_all.csv')
    # - link keys
    settings["paths"]["link_encounter_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'link_encounter_all.csv')
    settings["paths"]["link_recording_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'link_recording_all.csv')
    settings["paths"]["link_segment_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'link_segment_all.csv')
    settings["paths"]["signal_channel_key_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'signal_channel_key_all.csv')
    # - signal filepaths
    settings["paths"]["recording_filepaths_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'recording_filepaths_all.csv')  # signal filepaths for all subjects
    settings["paths"]["recording_filepaths_durations_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'recording_filepaths_durations_all.csv')  # signal filepaths and durations for all subjects
    settings["paths"]["recording_filepaths_root_all"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'recording_filepaths_root_all.txt')  # root folder for signal files (a prefix to the signal filepaths)
    # - processed variables
    settings["paths"]["all-signal-features-csv"] = str(Path(settings["paths"]["entire_dataset_proc_folder"]) / 'all-signal-features.csv')
    
    # Check if required directories exist, else ask for approval to create them
    dirs_to_check = [settings["paths"]["dataset_root_proc_folder"], settings["paths"]["entire_dataset_proc_folder"], settings["paths"]["derived_features_proc_folder"], settings["paths"]["preprocessing_proc_folder"]]
    ensure_directories_exist(dirs_to_check)
    
    # Save settings to JSON file
    with open(settings_file_path, "w") as f:
        json.dump(settings, f, indent=4)

    # End-matter
    print(f" - Settings file written to: {settings_file_path}")
    function_start_end_print()

if __name__ == "__main__":

    # Check whether expected number of inputs have been provided
    if len(sys.argv) != 4:
        print("Usage: python -m longitudinal_ecg_analysis.gen_dataset_settings <dataset_root_raw_folder> <dataset_root_proc_folder> <dataset_name>")
        sys.exit(1)

    # parse inputs
    dataset_root_raw_folder = sys.argv[1]
    dataset_root_proc_folder = sys.argv[2]
    dataset_name = sys.argv[3]

    # standardise dataset name
    dataset_name = standardise_name(dataset_name)

    # call function to generate dataset settings file
    _generate_dataset_settings_file(dataset_root_raw_folder, dataset_root_proc_folder, dataset_name)
