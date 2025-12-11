# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
curate_entire_dataset.py

Curates an entire dataset so that it can be analysed.
"""
import sys
import os
import pandas as pd
import numpy as np
import wfdb
import pyedflib
from pathlib import Path
from longitudinal_ecg_analysis.utils import load_dataset_settings, function_start_end_print, load_recording_filepath_root, get_recording_file_extension
from longitudinal_ecg_analysis.dataset_curators.curate_dataset_music import curate_dataset_music
from longitudinal_ecg_analysis.dataset_curators.curate_dataset_hh import curate_dataset_hh
from longitudinal_ecg_analysis.dataset_curators.curate_dataset_mcmed import curate_dataset_mcmed

def curate_the_entire_dataset(
    dataset_root_proc_folder: str,
    redo_everything: bool
):
    """Curate an entire dataset for analysis.

    Args:
        dataset_root_proc_folder (str): Path to folder containing the settings file (named "dataset_settings.json") and in which to save the curated dataset.
        redo_everything (bool): Indicates whether to redo all processing steps
        
    Returns:
        Writes curated data files to disk.
    """

    # Load dataset settings from file
    settings = load_dataset_settings(dataset_root_proc_folder, redo_everything)

    # Start-matter
    function_start_end_print()
    print(f"Curating entire '{settings['dataset_name']}' dataset")

    # Dataset-specific preparation
    print(f" - Performing dataset-specific data curation")
    perform_dataset_specific_data_curation(settings)
    
    # Derive additional standardised metricss
    print(f" - Deriving additional standardised metrics")
    derive_additional_standardised_metrics(settings)

    # Extract duration of each recording
    print(f" - Extracting recording durations")
    extract_recording_durations(settings)

    # End-matter
    print(f" - Finished curating entire '{settings['dataset_name']}' dataset")
    function_start_end_print()

    return


def perform_dataset_specific_data_curation(settings):

    # Check whether this has been done already:
    if Path(settings["paths"]["signal_channel_key_all"]).exists() & (not settings["redo_curation"]):
        print(f"   - Dataset-specific curation already done")
        return

    # Obtain dataset-specific preparation function
    print(f"   - Obtaining dataset-specific curation function")
    curation_function = obtain_dataset_specific_curation_function(settings)

    # Call the dataset-specific preparation function
    curation_function(settings)

    # Checking prepared dataset
    print(f"   - Checking dataset-specific preparation")
    check_dataset_specific_preparation(settings)

    return

    
def obtain_dataset_specific_curation_function(settings):

    # Find out whether a function is available to prepare this dataset
    dataset_name = settings['dataset_name']
    function_name = f"curate_dataset_{dataset_name}"

    # Try to get the function by name from the current module
    module = sys.modules[__name__]
    curation_function = getattr(module, function_name, None)

    # check whether a dataset-specific preparation function is available
    if curation_function is None:
        raise NotImplementedError(
            f"No curation function found for dataset '{dataset_name}'. "
            f"Expected a function named '{function_name}' in 'dataset_curators'."
    )

    return curation_function


def check_dataset_specific_preparation(settings):

    # Check that all required files have been created
    # - Specify required files
    req_files = [
        settings["paths"]["variables_subject_all"],
        settings["paths"]["variables_encounter_all"],
        settings["paths"]["link_encounter_all"],
        settings["paths"]["link_recording_all"],
        settings["paths"]["recording_filepaths_all"],
        settings["paths"]["recording_filepaths_root_all"],
        settings["paths"]["signal_channel_key_all"]
    ]
    # - Check that each one has been created
    for filepath in req_files:
        filepath = Path(filepath)
        if not filepath.exists():
            raise Exception(f"'{filepath.name}' not created in dataset-specific preparation.")

    return

def derive_additional_standardised_metrics(settings):
    """
    Derive additional standardised variables from routine metrics
    """

    # repeat for encounter and subject variables
    variable_types = ['subject', 'encounter']

    for variable_type in variable_types:

        # 'all' filepath
        # construct, e.g. as given by: settings["paths"]["variables_subject_all"], or settings["paths"]["variables_encounter_all"]
        key = f"variables_{variable_type}_all"
        all_filepath = settings["paths"].get(key)

        # load 'all' data from the csv given by 'all' filepath
        all_data = pd.read_csv(all_filepath)

        # derive additional metrics
        all_data = _derive_additional_metrics(all_data)

        # save back to same filepath
        all_data.to_csv(all_filepath, index=False)
    
    return

    
def _derive_additional_metrics(all_data):
    """
    Derive additional standardised variables from routine clinical metrics

    Parameters
    ----------
    all_data : ...

    Returns
    -------
    all_data : updated to include additional variables
    """

    if ('weight' in all_data.columns) & ('height' in all_data.columns) & ('bmi' not in all_data.columns):
        all_data['bmi'] = all_data['weight'] / (all_data['height'] ** 2)
        all_data['bmi'] = all_data['bmi'].round(1)
    
    if ('lvef' in all_data.columns) & ('lvef_under_35' not in all_data.columns):
        all_data['lvef_under_35'] = (all_data['lvef'] <= 35).astype(int)
    
    if ('nyha' in all_data.columns) & ('nyha_class_iii' not in all_data.columns):
        all_data['nyha_class_iii'] = (all_data['nyha'] == 3).astype(int)
    
    if ('arb' in all_data.columns) & ('ace_inhibitor' in all_data.columns) & ('arb_or_ace_inhibitor' not in all_data.columns):
        all_data['arb_or_ace_inhibitors'] = ((all_data['arb'] == 1) | (all_data['ace_inhibitor'] == 1)).astype(int)
    
    return all_data


def extract_recording_durations(settings):

    # check whether this has been done:
    if Path(settings["paths"]["recording_filepaths_durations_all"]).exists() & (not settings["redo_curation"]):
        print(f"   - Skipping as already done")
        return

    # load individual recording file paths
    rec_filepaths = pd.read_csv(settings["paths"]["recording_filepaths_all"])

    # load root filepath
    root_filepath = load_recording_filepath_root(settings, 'all')

    # go through each recording, and extract its duration
    rec_filepaths['duration'] = pd.NA
        
    def get_duration(row, root_filepath):
            
        # establish filepath of current recording
        curr_filetype = row['filetype']
        curr_rec_filepath = os.path.join(root_filepath, row['filepath'])
        root, ext = os.path.splitext(curr_rec_filepath)
        if ext:
            curr_rec_filepath_w_ext = curr_rec_filepath
        else:
            # add extension if required
            curr_ext = get_recording_file_extension(curr_filetype)
            curr_rec_filepath_w_ext = os.path.join(root_filepath, row['filepath'] + curr_ext)
        
        # check whether this filepath exists
        if not Path(curr_rec_filepath_w_ext).exists():
            print(row['filepath'])
            print(curr_rec_filepath)
            print(curr_rec_filepath_w_ext)
            print(Path(curr_rec_filepath_w_ext))
            raise Exception(f"No file found at: {curr_rec_filepath_w_ext}")
        
        # extract duration of current recording
        if row['filetype'] == 'WFDB':
            info = wfdb.rdheader(curr_rec_filepath)
            durn = (info.sig_len - 1) / info.fs  # in seconds
            return durn
        else:
            with pyedflib.EdfReader(curr_rec_filepath) as f:
                n_signals = f.signals_in_file
                fs = f.getSampleFrequency(0)   # sampling rate of first signal
                n_samples = f.getNSamples()[0] # number of samples in first signal
                durn = (n_samples-1) / fs  # in seconds
            
    print(f"   - This could take a while")        
    rec_filepaths['duration'] = rec_filepaths.apply(
        lambda row: get_duration(row, root_filepath), axis=1
    )

    # save updated rec filepaths with durations to file
    rec_filepaths.to_csv(settings["paths"]["recording_filepaths_durations_all"], index=False)

    return


if __name__ == "__main__":

    # Check whether expected number of inputs have been provided
    if (len(sys.argv) != 2) & (len(sys.argv) != 3):
        print("Usage: python -m longitudinal_ecg_analysis.curate_entire_dataset <dataset_root_proc_folder> [--redo_everything]")
        sys.exit(1)

    # parse inputs
    dataset_root_proc_folder = sys.argv[1]
    if len(sys.argv)==3:
        redo_everything = sys.argv[2]
    else:
        redo_everything = False

    # call function to curate the entire dataset
    curate_the_entire_dataset(dataset_root_proc_folder, redo_everything)
