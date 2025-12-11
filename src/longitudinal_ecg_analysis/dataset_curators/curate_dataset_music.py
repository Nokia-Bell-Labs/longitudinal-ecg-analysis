# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
curate_dataset_music.py

Curates the MUSIC (Sudden Cardiac Death in Chronic Heart Failure) dataset for analysis.
"""
import pandas as pd
from pathlib import Path
from longitudinal_ecg_analysis.utils import load_rec_enc_subj, create_signal_channel_key

def curate_dataset_music(settings):
    """Curate the MUSIC (Sudden Cardiac Death in Chronic Heart Failure) dataset for analysis.
    The dataset is available at: https://doi.org/10.13026/fa8p-he52

    Args:
        settings (dict): Dataset settings loaded from a settings file.
        
    Returns:
        None. Writes the prepared data to disk.
    """

    # Check whether dataset is as expected
    settings = check_music_dataset_files(settings)

    # Extract standardised variables
    clinical_metrics = extract_standard_dataset_variables(settings)

    # Create list of paths for ECG recording files
    file_paths = identify_ECG_recording_file_paths(settings)

    # Create signal channel key
    rel_sigs = ["X", "Y", "Z"]
    create_signal_channel_key(rel_sigs, settings["paths"]["signal_channel_key_all"])

    return


def check_music_dataset_files(settings):
    """
    Check the presence of required files and directories for the MUSIC dataset.

    This function verifies that the expected input files and folders exist in the
    directory specified by `settings["paths"]["input_dir"]`. It checks for:

    - A CSV file named 'subject-info.csv'
    - A folder named 'Holter_ECG'

    These are essential for processing the MUSIC dataset. If any of the required
    files or folders are missing, a FileNotFoundError is raised.

    Parameters:
        settings (dict): A dictionary containing file path settings. It must include
                         'input_dir' under 'settings["paths"]'.

    Returns:
        dict: The updated `settings` dictionary with paths for 'subj-info-csv' and
              'holter_ecg_folder' added to `settings["paths"]`.

    Raises:
        FileNotFoundError: If any required file or folder does not exist.
    """
    print("   - Checking dataset files")

    # Raw data

    # - Generate paths
    settings["paths"]["subj-info-csv"] = Path(settings["paths"]["dataset_root_raw_folder"]) / 'subject-info.csv'
    settings["paths"]["holter_ecg_folder"] = Path(settings["paths"]["dataset_root_raw_folder"]) / 'Holter_ECG'

    # - Check whether required files and folders exist
    items_to_check = [
        settings["paths"]["subj-info-csv"],
        settings["paths"]["holter_ecg_folder"]
    ]
    for item in items_to_check:
        if not item.exists():
            raise FileNotFoundError(f'File not found at {item}')
    
    return settings


def extract_standard_dataset_variables(settings):
    """
    Extract clinical, outcome and dataset variables in a standardised format

    Parameters
    ----------
    settings : a dict of settings, including settings["paths"]["subj-info-csv"] - the path of subject-info.csv

    Returns
    -------
        Writes the following prepared data files to disk:
        standard-clinical-metrics.csv : A CSV file containing clinical metrics for each subject.
        standard-outcome-variables.csv : A CSV file containing outcome variables for each subject. 
        standard-dataset-variables.csv : A CSV file containing variables describing the dataset.
    """

    # see whether this has already been done
    if (not settings["redo_curation"]) & Path(settings["paths"]["variables_segment_all"]).exists() & Path(settings["paths"]["variables_subject_all"]).exists() & Path(settings["paths"]["variables_encounter_all"]).exists():
        return

    # load raw data from CSV
    print("   - Loading raw tabular data")
    subjinfo = pd.read_csv(settings["paths"]["subj-info-csv"], sep=';')

    ### Subject-level variables

    print("   - Extracting initial subject-level variables")

    # setup clinical metrics dataframe
    subj_vars = pd.DataFrame()

    # re-format standardised variables
    subj_vars['subj_id'] = subjinfo["Patient ID"]
    subj_vars['age'] = pd.to_numeric(subjinfo['Age'], errors='coerce')
    subj_vars['gender'] = pd.to_numeric(subjinfo['Gender (male=1)'], errors='coerce')
    subj_vars['gender'] = subj_vars['gender'].replace(1,2)  # male as 2
    subj_vars['gender'] = subj_vars['gender'].replace(0,1)  # female as 1
    subj_vars['weight'] = pd.to_numeric(subjinfo['Weight (kg)'], errors='coerce')
    subj_vars['height'] = pd.to_numeric(subjinfo['Height (cm)'], errors='coerce') / 100  # convert from m to cm
    subj_vars['dbp'] = pd.to_numeric(subjinfo['Diastolic blood  pressure (mmHg)'], errors='coerce')
    subj_vars['sbp'] = pd.to_numeric(subjinfo['Systolic blood pressure (mmHg)'], errors='coerce')
    subj_vars['lvef'] = pd.to_numeric(subjinfo['LVEF (%)'], errors='coerce')
    subj_vars['prior_mi'] = pd.to_numeric(subjinfo['Prior Myocardial Infarction (yes=1)'], errors='coerce')
    subj_vars['nyha'] = pd.to_numeric(subjinfo['NYHA class'], errors='coerce')
    subj_vars['diabetes'] = pd.to_numeric(subjinfo['Diabetes (yes=1)'], errors='coerce')
    subj_vars['arb'] = pd.to_numeric(subjinfo['Angiotensin-II receptor blocker (yes=1)'], errors = 'coerce')
    subj_vars['ace_inhibitor'] = pd.to_numeric(subjinfo['ACE inhibitor (yes=1)'], errors = 'coerce')
    subj_vars['beta_blockers'] = pd.to_numeric(subjinfo['Betablockers (yes=1)'], errors = 'coerce')
    subj_vars['amiodarone'] = pd.to_numeric(subjinfo['Amiodarone (yes=1)'], errors = 'coerce')
    subj_vars['no_pvcs_24hr'] = pd.to_numeric(subjinfo['Number of ventricular premature beats in 24h'], errors = 'coerce')
    
    # Add in additional variables
    # - HF etiology
    subjinfo['hf_etiology'] = pd.to_numeric(subjinfo['HF etiology - Diagnosis'], errors='coerce')
    subj_vars['ischemic_dilated_cardiomyopathy'] = (subjinfo['hf_etiology'] == 2).astype(int)
    # - HF status
    subj_vars['hf'] = 1
    # - Heart rhythm
    subjinfo['sinusal_rhythm'] = pd.to_numeric(subjinfo['Holter  rhythm '], errors = 'coerce')
    subj_vars['sinusal_rhythm'] = (subjinfo['sinusal_rhythm'] == 0).astype(int)
    # - Holter recording
    subj_vars['holter_available'] = pd.to_numeric(subjinfo['Holter available'], errors='coerce')
    # - duration of follow-up
    # - Event indicator: 0 = survivor, 1 = other death, 3 = SCD, 6 = PFD
    followupt = pd.to_numeric(subjinfo['Follow-up period from enrollment (days)'], errors='coerce')
    subj_vars['survived'] = subjinfo['Cause of death']==0
    subj_vars['survived_followup'] = followupt
    subj_vars['other_death'] = subjinfo['Cause of death']==1
    subj_vars['other_death_followup'] = followupt
    subj_vars['sudden_cardiac_death'] = subjinfo['Cause of death']==3
    subj_vars['sudden_cardiac_death_followup'] = followupt
    subj_vars['pump_failure_death'] = subjinfo['Cause of death']==3
    subj_vars['pump_failure_death_followup'] = followupt
    
    # save to file
    subj_vars.to_csv(settings["paths"]["variables_subject_all"], index=False)

    ### Encounter-level variables and encounter link
    enc_vars = subj_vars.copy()
    # - encounter ID
    enc_vars['enc_id'] = [f'E{i}' for i in range(1, len(enc_vars) + 1)]
    enc_vars = enc_vars[['enc_id'] + [col for col in enc_vars.columns if col != 'enc_id']]
    # - create encounter link
    enc_link = enc_vars[['enc_id', 'subj_id']]
    # - drop subject ID
    enc_vars = enc_vars.drop(columns=['subj_id'])
    # - save to file
    enc_vars.to_csv(settings["paths"]["variables_encounter_all"], index=False)
    enc_link.to_csv(settings["paths"]["link_encounter_all"], index=False)

    ### Recording link
    filtered = enc_vars[enc_vars['holter_available'] == True]
    rec_link = filtered[['enc_id']].copy()
    rec_link['rec_id'] = [f'R{i}' for i in range(1, len(rec_link) + 1)]
    # - add availability of signals
    sigs = ['X', 'Y', 'Z']
    for curr_sig in sigs:
        rec_link[f"{curr_sig}_available"] = True
    # - save to file
    rec_link.to_csv(settings["paths"]["link_recording_all"], index=False)    
    
    
    return


def identify_ECG_recording_file_paths(settings):
    """
    Create file paths for ECG recordings and save them to a CSV.

    Parameters:
        settings (dict): Dictionary containing paths, including 'standard-clinical-metrics-csv', 'holter_ecg_folder' and 'signal-filepaths-csv'.

    Returns:
        pd.DataFrame: DataFrame with columns 'subj_id' and 'filepath'.
    """

    print("   - Creating list of paths for recording files")

    # obtain list of subjects
    rec_enc_subj = load_rec_enc_subj(settings)

    # specify root folder
    filepaths_root = Path(settings["paths"]["dataset_root_raw_folder"]) / "Holter_ECG"

    # obtain list of paths
    signal_filepaths = []
    for _, row in rec_enc_subj.iterrows():
        rec_id = row['rec_id']
        subj_id = row['subj_id']
        # curr_path = filepaths_root / subj_id
        signal_filepaths.append({
            'rec_id': rec_id, 
            'filepath': subj_id, 
            'filetype': 'WFDB',
        })

    # store paths to file
    df = pd.DataFrame(signal_filepaths)
    df.to_csv(settings["paths"]["recording_filepaths_all"], index=False)

    # store root path to file
    with open(settings["paths"]["recording_filepaths_root_all"], 'w') as f:
        f.write(str(filepaths_root))

    return df
