# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
gen_analysis_settings.py

Generates a JSON settings file for a given analysis. Can be run as a script
or imported as a module.
"""
import sys
import json
from pathlib import Path
from longitudinal_ecg_analysis.utils import obtain_custom_analysis_settings, standardise_name, ensure_directories_exist, load_dataset_settings, obtain_analysis_settings_file_path, function_start_end_print

def _generate_analysis_settings_file(
    dataset_root_proc_folder: str,
    analysis_name: str,
    do_demo=False,
):
    """Generate a JSON settings file for the analysis.

    Args:
        dataset_root_proc_folder (str): Path to folder to save settings, processed data, results, etc.
        analysis_name (str): Name of the analysis (e.g. attempt1).

    Returns:
        None. Writes the settings file to disk.
    """
    
    # Start-matter
    function_start_end_print()
    print(f"Generating settings file for the '{analysis_name}' analysis.")
  
    dataset_root_proc_folder = Path(dataset_root_proc_folder).resolve()

    # load dataset settings
    settings = load_dataset_settings(dataset_root_proc_folder)

    # add in generic analysis settings
    settings["analysis_name"] = analysis_name
    settings["redo_analysis"] = True
    # - Processed data folders and files
    settings["paths"]["analysis_proc_folder"] = str(dataset_root_proc_folder / analysis_name)
    settings["paths"]["analysis_files_folder"] = str(Path(settings["paths"]["analysis_proc_folder"]) / "analysis")
    settings["paths"]["rel_encs_csv"] = str(Path(settings["paths"]["analysis_files_folder"]) / "rel_encs.csv")
    settings["paths"]["variables_encounter_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "variables_encounter_analysis.csv")
    settings["paths"]["variables_subject_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "variables_subject_analysis.csv")
    settings["paths"]["link_encounter_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "link_encounter_analysis.csv")
    settings["paths"]["link_recording_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "link_recording_analysis.csv")
    settings["paths"]["link_segment_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "link_segment_analysis.csv")
    settings["paths"]["recording_filepaths_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "recording_filepaths_analysis.csv")
    settings["paths"]["recording_filepaths_root_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "recording_filepaths_root_analysis.csv")
    settings["paths"]["signal_channel_key_analysis"] = str(Path(settings["paths"]["analysis_files_folder"]) / "signal_channel_key_analysis.csv")
    settings["paths"]["predictor_response_table"] = str(Path(settings["paths"]["analysis_files_folder"]) / "predictor_response_table.csv")
    settings["paths"]["aggregate_encounter_features"] = str(Path(settings["paths"]["analysis_files_folder"]) / 'agg_enc_features.csv')
    # - results folder and files
    settings["paths"]["analysis_res_folder"] = str(Path(settings["paths"]["analysis_proc_folder"]) / "results")
    settings["paths"]["tableone"] = str(Path(settings["paths"]["analysis_res_folder"]) / "tableone.csv")

    # Check if required output directories exist, else ask for approval to create them
    dirs_to_check = [
        settings["paths"]["analysis_proc_folder"],
        settings["paths"]["analysis_files_folder"],
        settings["paths"]["analysis_res_folder"],
    ]
    ensure_directories_exist(dirs_to_check)

    # Add in analysis settings for this particular analysis
    custom_settings = obtain_custom_analysis_settings(dataset_root_proc_folder, settings["dataset_name"], analysis_name, do_demo)

    settings["analysis"] = custom_settings

    # Set default analysis settings
    settings["analysis"].setdefault("max_encs", -1)
    settings["analysis"].setdefault("max_subjs", -1)
    settings["analysis"].setdefault("sigs_to_exclude", [])

    # Save settings JSON
    settings_file_path = obtain_analysis_settings_file_path(dataset_root_proc_folder, analysis_name)
    with open(settings_file_path, "w") as f:
        json.dump(settings, f, indent=4)

    # End-matter
    print(f" - Settings file written to: {settings_file_path}")
    function_start_end_print()


if __name__ == "__main__":

    # Check whether expected number of inputs have been provided
    if len(sys.argv) < 3:
        print("Usage: python -m longitudinal_ecg_analysis.gen_dataset_settings <dataset_root_proc_folder> <analysis_name> <do_demo>")
        sys.exit(1)

    # parse inputs
    dataset_root_proc_folder = sys.argv[1]
    analysis_name = sys.argv[2]

    # insert 'do_demo' variable if required
    if len(sys.argv) < 4:
        if analysis_name == "demo_analysis":
            do_demo = True
        else:
            do_demo = False
    else:
        do_demo = sys.argv[3]

    # standardise analysis name
    analysis_name = standardise_name(analysis_name)

    # call function to generate analysis settings file
    _generate_analysis_settings_file(dataset_root_proc_folder, analysis_name, do_demo)
