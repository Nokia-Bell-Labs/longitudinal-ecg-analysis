# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
utils.py

Common utilities
"""
import json
import os
import pandas as pd
from pathlib import Path
import warnings

def obtain_dataset_settings_file_path(dataset_root_proc_folder):
    """Obtain path of dataset settings file."""

    settings_filename = "dataset_settings.json"
    processing_path = Path(dataset_root_proc_folder).resolve()
    settings_file_path = processing_path / settings_filename

    return settings_file_path


def obtain_analysis_settings_file_path(dataset_root_proc_folder, analysis_name):
    """Obtain path of analysis settings file."""

    settings_filename = "analysis_settings.json"
    processing_path = Path(dataset_root_proc_folder).resolve()
    settings_file_path = processing_path / analysis_name / settings_filename

    return settings_file_path


def load_dataset_settings(processing_dir, redo_everything=False):
    """Load dataset settings from file."""

    # Obtain settings file path
    settings_file_path = obtain_dataset_settings_file_path(processing_dir)

    # Check settings file exists at specified path
    if not settings_file_path.exists():
        raise FileNotFoundError(f"No settings file found at: {settings_file_path}. Use `gen_dataset_settings` to generate a settings file.")

    # Load settings
    with open(settings_file_path, "r") as f:
        settings = json.load(f)
    
    # Override redo settings
    if redo_everything:
        for key in settings.keys():
            ## ask whether the name of the current key includes the string 'redo'
            if "redo" in key:
                settings[key] = True

    return settings


def load_analysis_settings(processing_dir, analysis_name):
    """Load analysis settings from file."""

    # Load analysis settings
    settings_file_path = obtain_analysis_settings_file_path(processing_dir, analysis_name)

    # Check settings file exists at specified path
    if not settings_file_path.exists():
        raise FileNotFoundError(f"No analysis settings file found at: {settings_file_path}. Use `gen_analysis_settings` to generate the file.")

    # Load settings
    with open(settings_file_path, "r") as f:
        settings = json.load(f)

    return settings


def load_enc_data(settings):

    enc_filepath = settings["paths"]["variables_encounter_all"]

    enc_data = pd.read_csv(enc_filepath)

    return enc_data


def load_rel_encs(settings):

    rel_encs_filepath = settings["paths"]["rel_encs_csv"]

    rel_encs = pd.read_csv(rel_encs_filepath)

    return rel_encs


def ask_approval_to_create_dir(path: Path) -> bool:
    """Ask user for approval before creating a directory."""
    while True:
        response = input(f"Directory '{path}' does not exist. Create it? (y/n): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            print("Directory creation aborted by user.")
            return False
        else:
            print("Please enter 'y' or 'n'.")

def function_start_end_print():

    print('~~~~~~~~~~~~~~~~~~~')


def load_rec_enc_subj(settings, include_filenames=False):
    """Load list of recordings and corresponding encounter IDs and subject IDs."""

    # load link dataframes
    link_rec = pd.read_csv(settings["paths"]["link_recording_all"])  # contains rec_id and enc_id
    link_enc = pd.read_csv(settings["paths"]["link_encounter_all"])  # contains enc_id and subj_id

    # Merge the two DataFrames on 'enc_id' using an inner join
    rec_enc_subj = pd.merge(link_rec, link_enc, on='enc_id', how='inner')

    # Extract only the desired columns
    rel_cols = ['rec_id', 'enc_id', 'subj_id']
    rec_enc_subj = rec_enc_subj[rel_cols]

    # add in filename if required
    if include_filenames:
        link_rec = pd.read_csv(settings["paths"]["variables_encounter_all"])  # contains enc_id and filename
        link_rec = link_rec[['enc_id', 'filename']]
        rec_enc_subj = pd.merge(rec_enc_subj, link_rec, on='enc_id', how='inner')

    return rec_enc_subj


def standardise_name(init_name):
    """Standardise a name (e.g. of a dataset or an analysis)"""

    standard_name = init_name.lower()

    standard_name = standard_name.replace('-','')

    return standard_name


def ensure_directories_exist(dir_paths):
    """
    Ensures each directory in dir_paths exists. 
    If a directory does not exist, it asks for approval to create it.
    If approval is denied, exits early.
    """
    for curr_path in dir_paths:
        curr_path = Path(curr_path)
        if not curr_path.exists():
            if not ask_approval_to_create_dir(curr_path):
                print(" - Exiting without creating settings file.")
                return
            curr_path.mkdir(parents=True)

def obtain_custom_analysis_settings(dataset_root_proc_folder, dataset_name, analysis_name, do_demo):

    # specify custom analysis settings filepath
    settings_filename = analysis_name + "_specific_settings.json"
    processing_path = Path(dataset_root_proc_folder).resolve()
    settings_file_path = processing_path / settings_filename

    # create custom analysis settings file if doing the demo
    if do_demo:
        # Define default demo settings
        demo_settings = {
                "max_subjs": 10,
                "max_encs": 1,
                "inclusion_criteria": {},
            }
        # Add in dataset-specific dataset settings
        if dataset_name == 'music':
            demo_settings.update({
                "inclusion_criteria": {
                    "holter_available": 1,
                    "sinusal_rhythm": 1,
                    "prior_mi": 0
                },
                "sigs_to_exclude": ["Z", "Y"]
            })
        elif dataset_name == 'hh':
            # I think this one already has any patients without signal data excluded
            demo_settings.update({
                "max_subjs": 50,
                "inclusion_criteria": {
                    "af": "no AF",
                },
                "sigs_to_exclude": ["ECG I", "ECG V1", "ECG V2", "ECG V3", "ECG V4", "ECG V5", "ECG V6", "ACC X", "ACC Y", "ACC Z"]
            })
        elif dataset_name == 'mcmed':
            demo_settings.update({
                "sigs_to_exclude": ["Pleth", "Resp"],
                "inclusion_criteria": {
                    "II_duration": 1,  # in minutes
                },
            })
        else:
            print(f"No dataset-specific settings found for {dataset_name} dataset")

        # Write the JSON file
        with open(settings_file_path, 'w') as f:
            json.dump(demo_settings, f, indent=4)

        print(f" - Demo analysis settings file created.")

    # Check settings file exists at specified path
    if not settings_file_path.exists():
        raise FileNotFoundError(f"No analysis-specific settings file found at: {settings_file_path}. <insert details of how to generate>")

    # Load settings
    with open(settings_file_path, "r") as f:
        settings = json.load(f)

    return settings


def identify_predictor_response_variables(var_list):

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the JSON file
    config_path = os.path.join(script_dir, "variable_categories.json")

    # Load the JSON config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get signal prefixes and convert to usable form (e.g., 'X_' instead of 'X')
    signal_prefixes = [prefix + '_' for prefix in config.get("signal_prefixes", [])]
    variable_map = config.get("variables", {})

    predictors = []
    responses = []
    time_to_responses = []
    variable_types = {}

    for var in var_list:

        # Signal-derived predictor
        if any(var.startswith(prefix) for prefix in signal_prefixes):
            predictors.append(var)
            continue
        
        # see whether this is a time to response
        if var.endswith("_followup"):
            time_to_responses.append(var)
            var_type = "numerical"
            variable_types[var] = var_type
            continue
        
        # extract info on this variable
        var_info = variable_map.get(var)
        if not var_info:
            print(f"'{var}' variable needs defining in 'variable_categories.json'")
            continue
        category = var_info.get("type")
        var_type = var_info.get("var_type")

        # record category type
        if category == "predictor":
            predictors.append(var)
        elif category == "response":
            responses.append(var)
        elif category != "dataset":
            warnings.warn(f"Variable '{var}' not defined in `variable_categories.json`")
        
        variable_types[var] = var_type

    return {
        "predictors": predictors,
        "responses": responses,
        "time_to_responses": time_to_responses,
        "variable_types": variable_types
    }


def create_signal_channel_key(rel_sigs, save_filepath):

    # create a key to link between the signal names in the dataset files, and the standardised names
    sig_key = define_universal_signal_key()

    rel_sig_key = {k: sig_key[k] for k in rel_sigs if k in sig_key}
    
    # Save to JSON file
    with open(save_filepath, "w") as f:
        json.dump(rel_sig_key, f, indent=4)

    return


def define_universal_signal_key():

    # Define a key linking the signal names in a dataset (left) to standardised names (right)

    sig_key = {
        "ECG I": "ecgI",
        "ECG II": "ecgII",
        "ECG V1": "ecgV1",
        "ECG V2": "ecgV2",
        "ECG V3": "ecgV3",
        "ECG V4": "ecgV4",
        "ECG V5": "ecgV5",
        "ECG V6": "ecgV6",
        "ACC X": "accX",
        "ACC Y": "accY",
        "ACC Z": "accZ",
        "X": "ecgX",
        "Y": "ecgY",
        "Z": "ecgZ",
        "Pleth": "ppg",
        "II": "ecgII",
        "Resp": "resp",
    }

    return sig_key


def standardise_sig_name(curr_sig):

    sig_key = define_universal_signal_key()
    standard_sig = sig_key[curr_sig]

    return standard_sig


def load_recording_filepath_root(settings, type):

    if type == 'analysis':
        txt_file_path = settings["paths"]["recording_filepaths_root_analysis"]
    elif type == 'all':
        txt_file_path = settings["paths"]["recording_filepaths_root_all"]

    with open(txt_file_path, 'r', encoding='utf-8') as f_in:
        filepath_root = f_in.read()

    return filepath_root


def identify_recs(settings, type):
    """Obtain details of recordings to be analysed: (i) filepath; (ii) signals in recording

    """
    
    if type == 'analysis':
        rec_link_file_path = settings["paths"]["link_recording_analysis"]
        rec_filepaths_file_path = settings["paths"]["recording_filepaths_analysis"]
    elif type == 'all':
        rec_link_file_path = settings["paths"]["link_recording_all"]
        rec_filepaths_file_path = settings["paths"]["recording_filepaths_all"]

    # load recording IDs to be analysed
    rec_link = pd.read_csv(rec_link_file_path)
    
    # load recording filepaths
    rec_filepaths = pd.read_csv(rec_filepaths_file_path)
    
    # merge
    recs = pd.merge(rec_link, rec_filepaths, on='rec_id', how='inner')
    
    return recs


def identify_req_signals(settings):

    req_signals = []
    # go through each of the inclusion criteria
    for key in settings["analysis"]["inclusion_criteria"].keys():
        # search for inclusion criteria such as "II_available"
        if "_duration" in key:
            # extract signal types such as "II"
            sig_type = key.split('_')[0]
            req_signals.append(sig_type)

    return req_signals


def get_recording_file_extension(filetype):

    if filetype == "WFDB":
        ext = ".hea"
    elif filetype == "EDF":
        ext = ".edf"

    return ext