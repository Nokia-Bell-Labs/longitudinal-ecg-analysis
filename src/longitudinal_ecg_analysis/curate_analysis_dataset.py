# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
curate_analysis_dataset.py

Curates a subset of a dataset for a particular analysis.
"""
import sys
import pandas as pd
import json
from pathlib import Path
from longitudinal_ecg_analysis.utils import standardise_name, load_analysis_settings, load_enc_data, load_rel_encs, function_start_end_print, create_signal_channel_key, identify_req_signals

def curate_the_analysis_dataset(
    dataset_root_proc_folder: str,
    analysis_name: str
):
    """Curate a subset of a dataset for a particular analysis.

    Args:
        dataset_root_proc_folder (str): Path to folder containing the settings file (named "dataset_settings.json") and in which to save the curated dataset.
        analysis_name (str): Name of the analysis (e.g. attempt1).
        
    Returns:
        Writes the following curated data files to disk:
        ...
    """

    # Load analysis settings from file
    settings = load_analysis_settings(dataset_root_proc_folder, analysis_name)

    # Start-matter
    function_start_end_print()
    print(f"Curating subset of '{settings['dataset_name']}' dataset for '{settings['analysis_name']}' analysis")

    # Identify encounters to be included in the analysis
    print(f" - Identifying encounters to be included")
    create_list_rel_encs(settings)

    # Extract reduced set of variables and metadata from entire curated dataset for these encounters
    print(f" - Extracting reduced subset for this analysis")
    extract_subset_for_analysis(settings)

    # 
    
    # End-matter
    print(f" - Finished curating subset of '{settings['dataset_name']}' dataset for '{settings['analysis_name']}' analysis")
    function_start_end_print()

    return


def apply_inclusion_criteria(df, criteria: dict):
    for col, required_value in criteria.items():
        if 'duration' in col:
            print(f"     - criterion: {col} >= {required_value}")
            df = df[df[col] >= required_value]
        else:
            print(f"     - criterion: {col} == {required_value}")
            df = df[df[col] == required_value]
    return df.reset_index(drop=True)


def create_list_rel_encs(settings):
    """Create a list of relevant encounters to be included in the analysis.

    Args:
        settings : ...
        
    Returns:
        Writes the following file to disk:
        rel-subjs.csv : A CSV file containing a single column listing the subject IDs for each subject included in the analysis.
    """
    
    # load all subject variables
    print(f"   - Loading encounter data")
    enc_data = load_enc_data(settings)
    print(f"     - with    {len(enc_data)} encounters loaded")

    # Narrowing down to only those encounters which meet the inclusion criteria:
    print(f"   - Identifying encounters which meet inclusion criteria:")
    enc_data = apply_inclusion_criteria(enc_data, settings["analysis"]["inclusion_criteria"])
    print(f"     - leaving {len(enc_data)} encounters included")

    # Narrowing down to the maximum number of subjects to be analysed
    if settings["analysis"]["max_subjs"] != -1:
        print(f'   - Narrowing down to {settings["analysis"]["max_subjs"]} subjects')
        # Import subject IDs
        link_enc = pd.read_csv(settings["paths"]["link_encounter_all"])
        # Merge subj id and enc id
        enc_data_with_subj = enc_data.merge(link_enc, on='enc_id', how='left')
        # Step 2: Get the first N unique subj_ids from enc_data (not from all data)
        subj_ids_to_keep = enc_data_with_subj['subj_id'].drop_duplicates().head(settings["analysis"]["max_subjs"])
        # Step 3: Filter the merged data to only those N subjects
        enc_data_filtered = enc_data_with_subj[enc_data_with_subj['subj_id'].isin(subj_ids_to_keep)]
        # Reset index
        enc_data_filtered.reset_index(drop=True, inplace=True)
        # store in enc_data
        enc_data = enc_data_filtered
        print(f"     - leaving {len(enc_data)} encounters included")
    
    # Narrowing down to the maximum number of encounters to be analysed per subject
    if settings["analysis"]["max_encs"] != -1:
        print(f'   - Narrowing down to {settings["analysis"]["max_encs"]} encounter(s) per subject')
        # Group by subj_id and take the first max_encs rows per subject
        enc_data = enc_data.groupby('subj_id').head(settings["analysis"]["max_encs"]).reset_index(drop=True)
        print(f"     - leaving {len(enc_data)} encounters included")
        
    # extract list of relevant encounters
    rel_encs = enc_data[['enc_id']]

    # save relevant subjects to file
    rel_encs.to_csv(settings["paths"]["rel_encs_csv"], index=False)
    
    return


def extract_subset_for_analysis(settings):

    ## variables_encounter

    # load relevant encounters
    rel_encs = load_rel_encs(settings)

    # load encounter data
    filepath_all = settings["paths"]["variables_encounter_all"]
    df = pd.read_csv(filepath_all)

    # retain only relevant encounters
    df = df[df['enc_id'].isin(rel_encs['enc_id'])]

    # drop irrelevant variables
    cols_to_drop = ["filename", "rec_type", "device_type", "Visit_no", "visit_id"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # store to file
    df.to_csv(settings["paths"]["variables_encounter_analysis"], index=False)

    ## encounter link
    
    # load encounter link
    filepath_all = settings["paths"]["link_encounter_all"]
    link_enc = pd.read_csv(filepath_all)

    # identify relevant encounter link data
    rel_link_enc = link_enc[link_enc['enc_id'].isin(rel_encs['enc_id'])]

    # store to file
    rel_link_enc.to_csv(settings["paths"]["link_encounter_analysis"], index=False)
    
    ## variables_subject

    # identify relevant subjects
    rel_subjs = rel_link_enc['subj_id']
    
    # load subject data
    filepath_all = settings["paths"]["variables_subject_all"]
    subj_df = pd.read_csv(filepath_all)
    
    # retain only relevant subjects
    subj_df = subj_df[subj_df['subj_id'].isin(rel_subjs)]

    # store to file
    subj_df.to_csv(settings["paths"]["variables_subject_analysis"], index=False)

    ## link_recording_analysis

    # load recording link
    filepath_all = settings["paths"]["link_recording_all"]
    link_rec = pd.read_csv(filepath_all)

    # identify relevant encounter link data
    print(f"   - Identifying recordings which correspond to these encounters")
    rel_link_rec = link_rec[link_rec['enc_id'].isin(rel_encs['enc_id'])]
    print(f"     - leaving {len(rel_link_rec)} out of {len(link_rec)} recordings included")
    
    # narrow down to only those recordings containing the required signal(s) of interest:
    req_signals = identify_req_signals(settings)
    print(f"   - Identifying recordings which include the required signals: {req_signals}")
    # Start with a boolean mask that assumes all rows should be kept (True)
    recs_to_keep = pd.Series(True, index=rel_link_rec.index)
    # Loop through the required signals and update the mask
    for signal in req_signals:
        # Construct the column name dynamically
        col_name = f"{signal}_available"
        # Update the mask: keep a row only if it was already True AND the signal is available
        recs_to_keep = recs_to_keep & rel_link_rec[col_name]
    # Filter the DataFrame using the final mask
    rel_link_rec = rel_link_rec[recs_to_keep]
    print(f"     - leaving {len(rel_link_rec)} out of {len(recs_to_keep)} recordings included")

    # store to file
    rel_link_rec.to_csv(settings["paths"]["link_recording_analysis"], index=False)

    ## recording_filepaths_analysis

    # load all recording filepaths
    filepath_all = settings["paths"]["recording_filepaths_all"]
    all_filepaths = pd.read_csv(filepath_all)

    # identify relevant recording filepaths
    rel_filepaths = all_filepaths[all_filepaths['rec_id'].isin(rel_link_rec['rec_id'])]

    # store to file
    rel_filepaths.to_csv(settings["paths"]["recording_filepaths_analysis"], index=False)

    ## recording_filepaths_root

    # copy across
    with open(settings["paths"]["recording_filepaths_root_all"], 'r', encoding='utf-8') as f_in:
        contents = f_in.read()
    with open(settings["paths"]["recording_filepaths_root_analysis"], 'w', encoding='utf-8') as f_out:
        f_out.write(contents)
    
    ## signal_channel_key
    # - load all signals as keys in 'signal_channel_key_all'
    with open(settings["paths"]["signal_channel_key_all"], "r") as f:
        signal_channel_key_all = json.load(f)
    all_sigs = list(signal_channel_key_all.keys())
    # - find set of signals to be included in the analysis
    sigs_to_exclude = settings['analysis']['sigs_to_exclude']
    sigs_to_include = [sig for sig in all_sigs if sig not in sigs_to_exclude]
    create_signal_channel_key(sigs_to_include, settings["paths"]["signal_channel_key_analysis"])
    
    return


if __name__ == "__main__":
    
    # Check whether expected number of inputs have been provided
    if len(sys.argv) != 3:
        print("Usage: python -m longitudinal_ecg_analysis.curate_analysis_dataset <dataset_root_proc_folder> <analysis_name>")
        sys.exit(1)

    # parse inputs
    dataset_root_proc_folder = sys.argv[1]
    analysis_name = sys.argv[2]

    # standardise analysis name
    analysis_name = standardise_name(analysis_name)

    # call function to curate the analysis-specific dataset
    curate_the_analysis_dataset(dataset_root_proc_folder, analysis_name)
