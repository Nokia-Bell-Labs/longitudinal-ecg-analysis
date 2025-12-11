# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
curate_dataset_mcmed.py

Curates the MC-MED dataset for analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from longitudinal_ecg_analysis.utils import create_signal_channel_key

def curate_dataset_mcmed(settings):
    """Curate the MC-MED dataset for analysis.
    This is a freely available dataset, available at: https://doi.org/10.13026/xgx1-7x47

    Args:
        settings (dict): Dataset settings loaded from a settings file.
        
    Returns:
        None. Writes the prepared data to disk.
    """

    # Check whether dataset is as expected, and update settings with raw data paths
    settings = check_mcmed_dataset_files(settings)

    # Extract standardised variables
    rec_link_with_rec_id_orig = extract_standard_dataset_variables(settings)

    # Create list of paths for waveform recording files
    identify_waveform_recording_file_paths(rec_link_with_rec_id_orig, settings)

    # Create signal channel key
    rel_sigs = ["Pleth", "Resp", "II"]
    create_signal_channel_key(rel_sigs, settings["paths"]["signal_channel_key_all"])

    return


def check_mcmed_dataset_files(settings):
    """
    Check the presence of required files and directories for the MC-MED dataset.

    This function verifies that the expected input files and folders exist in the
    directory specified by `settings["paths"]["dataset_root_raw_folder"]`. It checks for:

    - A CSV file named 'visits.csv'
    - A CSV file named 'waveform_summary.csv'
    - A folder named 'waveforms'

    These are essential for processing the dataset. If any of the required
    files or folders are missing, a FileNotFoundError is raised.

    Parameters:
        settings (dict): A dictionary containing file path settings. It must include
                         "dataset_root_raw_folder" under 'settings["paths"]'.

    Returns:
        dict: The updated `settings` dictionary with paths for the necessary
              files and folders added.

    Raises:
        FileNotFoundError: If any required file or folder does not exist.
    """

    print("   - Checking dataset files")

    # Raw data

    # - Generate paths of raw data files and folders
    settings["paths"]["visits_file_path"] = Path(settings["paths"]["dataset_root_raw_folder"]) / "visits.csv"
    settings["paths"]["waveform_summary_file_path"] = Path(settings["paths"]["dataset_root_raw_folder"]) / "waveform_summary.csv"
    settings["paths"]["waveform_folder"] = Path(settings["paths"]["dataset_root_raw_folder"]) / "waveforms"
    
    # - Check whether required files and folders exist
    items_to_check = [
        settings["paths"]["visits_file_path"],
        settings["paths"]["waveform_summary_file_path"],
        settings["paths"]["waveform_folder"],
    ]
    for item in items_to_check:
        if not item.exists():
            raise FileNotFoundError(f'File not found at {item}')
    
    print("     - All required files found")
    
    return settings


def extract_standard_dataset_variables(settings):
    """
    Extract dataset variables in a standardised format

    Parameters
    ----------
    settings : a dict of settings

    Returns
    -------
        Writes the following prepared data files to disk:
        ...
    """

    # see whether this has already been done
    if (not settings["redo_curation"]) & Path(settings["paths"]["variables_segment_all"]).exists() & Path(settings["paths"]["variables_subject_all"]).exists() & Path(settings["paths"]["variables_encounter_all"]).exists():
        return

    # load raw visit metadata from CSV
    print("   - Loading raw tabular data")
    csv_data = pd.read_csv(settings["paths"]["visits_file_path"], dtype={'Arrival_time': str})

    # reformat variables
    var_info = create_var_info()
    csv_data = reformat_variables(csv_data, var_info)

    # load waveform summary data from CSV
    print("   - Loading waveform summary data")
    waveform_data = pd.read_csv(settings["paths"]["waveform_summary_file_path"])
    waveform_data = waveform_data.rename(columns={"CSN": "visit_id"})

    # merge visits metadata and waveform summary data
    csv_data = merge_df_waves_into_df(waveform_data, csv_data)

    ## Exclude subjects

    # - Exclude subjects with no waveform data
    csv_data["signal_recorded"] = (
        (csv_data["II_duration"] > 0) |
        (csv_data["Pleth_duration"] > 0) |
        (csv_data["Resp_duration"] > 0)
    )
    csv_data = csv_data[csv_data["signal_recorded"]]
    print(f"       - Excluded {sum(~csv_data['signal_recorded'])} out of {len(csv_data)} encounters because no signal recorded")
    csv_data = csv_data.drop(columns=["signal_recorded"])

    # - Exclude subjects which didn't complete ED assessment / standard pathway
    excluded_dispositions = [
        "ED Only - Left Without Being Seen",
        "ED Only - Erroneous Registration",
        "Left Against Medical Advice"
    ]
    csv_data["completed_ED_assessment"] = ~csv_data["DC_dispo"].isin(excluded_dispositions) # Set 'completed_ED_assessment' to True for any row NOT in the excluded list
    print(f"       - Excluded {sum(~csv_data['completed_ED_assessment'])} out of {len(csv_data)} encounters because ED assessment not completed")
    csv_data = csv_data[csv_data["completed_ED_assessment"]]
    csv_data = csv_data.drop(columns=["completed_ED_assessment"])
    
    ### Subject-level variables
    print("   - Extracting initial subject-level variables")

    subj_vars = csv_data.copy()
    
    # Remove unwanted variables
    subj_vars = subj_vars.drop(columns=["visit_id", "Visit_no", "Means_of_arrival", "temp", "heart_rate", "resp_rate",
       "spo2", "sbp", "dbp", "Triage_acuity", "CC", "ED_dispo", "Hours_to_next_visit", "Dispo_class_next_visit", "ED_LOS",
       "hosp_los", "DC_dispo", "Admit_service", "Dx_ICD9", "Dx_ICD10", "Dx_name", "Arrival_time", "Roomed_time", "Dispo_time",
       "Admit_time", "Departure_time", "age", "Days_until_next_visit", "arrival_time_dt", "II_duration", "Pleth_duration", "Resp_duration", "II_n_segments", "Pleth_n_segments", "Resp_n_segments"])

    # Keep only first row per subject
    subj_vars = subj_vars.drop_duplicates(subset="subj_id", keep="first")
    
    # save to file
    subj_vars.to_csv(settings["paths"]["variables_subject_all"], index=False)
    print(f"     - Saved data for {len(subj_vars)} subjects")
    
    ### Encounter-level variables and encounter link
    print("   - Extracting initial encounter-level variables")

    enc_vars = csv_data.copy()

    # - add encounter ID
    enc_vars['enc_id'] = [f'E{i}' for i in range(1, len(enc_vars) + 1)]
    enc_vars = enc_vars[['enc_id'] + [col for col in enc_vars.columns if col != 'enc_id']]
    # - create encounter link
    enc_link = enc_vars[['enc_id', 'subj_id']]
    # - add encounter-level outcomes
    #  - triage acuity (1 to 5)
    enc_vars["esi"] = enc_vars["Triage_acuity"]
    enc_vars["esi"] = enc_vars["esi"].str.split('-').str[0].astype('Int64')
    #  - ED disposition
    admitted_categories = ['Inpatient', 'ICU', 'Observation']
    enc_vars["admitted"] = enc_vars["ED_dispo"].isin(admitted_categories)
    #  - Hospital survival
    died_categories = ["Expired", "Expired to be Readmitted as Donor", "Expired. Organ Harvest Complete"]
    enc_vars["survived_hospital"] = ~enc_vars["DC_dispo"].isin(died_categories)
    #   - Multiple ED visits
    enc_vars['visited_ED_previously'] = enc_vars['Visit_no']>1
    enc_vars['visited_ED_subsequently'] = enc_vars['Visit_no']<enc_vars['no_visits']
    #   - Diagnosis categories
    enc_vars["icd10_diagnosis"] = enc_vars['Dx_ICD10'].fillna('Unknown')
    cardiac_pattern = r'^I(?:1[1-9]|[2-5][0-9])'  # True if the ICD10 code includes any of the strings I11, I12, I13, ..., or I59. Otherwise False.
    enc_vars['cardiac_diagnosis'] = enc_vars['icd10_diagnosis'].str.contains(cardiac_pattern, regex=True)
    cardiovascular_pattern = r'^I(?:1[1-9]|[2-9][0-9])|^I99'  # True if the ICD10 code includes any of the strings I11, I12, I13, ..., or I99. Otherwise False.
    enc_vars['cardiovascular_diagnosis'] = enc_vars['icd10_diagnosis'].str.contains(cardiovascular_pattern, regex=True)
    mi_pattern = r'^I(?:21|22)'  # True if the ICD10 code includes any of the strings I21 or I22. Otherwise False.
    enc_vars['mi_diagnosis'] = enc_vars['icd10_diagnosis'].str.contains(mi_pattern, regex=True)
    subs_mi_pattern = r'^I22'  # True if the ICD10 code includes the string I22. Otherwise False.
    enc_vars['subsequent_mi_diagnosis'] = enc_vars['icd10_diagnosis'].str.contains(subs_mi_pattern, regex=True)
    #   - Past and Future visits and admissions
    print("     - including information on past and future visits and admissions (which could take a while)")
    enc_vars = enc_vars.sort_values(['subj_id', 'Visit_no'])
    enc_vars = enc_vars.groupby('subj_id', group_keys=False).apply(compute_past_future_flags)

    # grab data for recording linl
    rec_link = enc_vars[['enc_id', 'II_n_segments', 'Pleth_n_segments', 'Resp_n_segments']].copy()

    # - remove unwanted variables
    enc_vars = enc_vars.drop(columns=["no_visits",
                                      "gender",
                                      "race",
                                      "ethnicity",
                                      "Dx_ICD10",
                                      "subj_id",
                                      "Triage_acuity",
                                      "CC",
                                      "ED_dispo",
                                      "Hours_to_next_visit",
                                      "Dispo_class_next_visit",
                                      "ED_LOS",
                                      "DC_dispo",
                                      "Admit_service",
                                      "Dx_ICD9",
                                      "Dx_name",
                                      "Arrival_time",
                                      "Roomed_time",
                                      "Dispo_time",
                                      "Admit_time",
                                      "Departure_time",
                                      "Days_until_next_visit",
                                      "II_n_segments",
                                      "Pleth_n_segments",
                                      "Resp_n_segments",
                                      "Visit_no",
                                      "Means_of_arrival",
                                      "arrival_time_dt"])
    
    # - save to file
    enc_vars.to_csv(settings["paths"]["variables_encounter_all"], index=False)
    enc_link.to_csv(settings["paths"]["link_encounter_all"], index=False)
    print(f"     - Saved data for {len(enc_vars)} encounters")

    ### Recording link
    print("   - Extracting recording link")
    # - started a few lines above

    # Step 1: calculate total number of segments per enc_id
    rec_link['total_segments'] = rec_link[['II_n_segments', 'Pleth_n_segments', 'Resp_n_segments']].sum(axis=1).astype(int)

    # Step 2: repeat rows by total number of segments
    df_expanded = rec_link.loc[rec_link.index.repeat(rec_link['total_segments'])].copy()

    # Step 3: assign rec_id within each enc_id group
    df_expanded['rec_id_cum'] = df_expanded.groupby('enc_id').cumcount() + 1

    # Step 4: assign signal type per segment
    def assign_signal_type(row):
        rec_id_cum = row['rec_id_cum']
        ii_end = row['II_n_segments']
        pleth_end = ii_end + row['Pleth_n_segments']
    
        if rec_id_cum <= ii_end:
            return 'II'
        elif rec_id_cum <= pleth_end:
            return 'Pleth'
        else:
            return 'Resp'

    df_expanded['signal_type'] = df_expanded.apply(assign_signal_type, axis=1)

    # Step 5: create boolean columns for availability (recordings each contain one and only one signal)
    df_expanded['II_available'] = df_expanded['signal_type'] == 'II'
    df_expanded['Pleth_available'] = df_expanded['signal_type'] == 'Pleth'
    df_expanded['Resp_available'] = df_expanded['signal_type'] == 'Resp'

    # Step 6: insert rec_id_orig, which is sequential from 1 for each signal and each enc_id, and refers to the rec ID in the waveform filenames
    df_expanded['rec_id_orig'] = (
        df_expanded
        .groupby(['enc_id', 'signal_type'])
        .cumcount() + 1
    )

    # Drop helper columns
    df_expanded = df_expanded.drop(columns=['total_segments', 'signal_type', 'II_n_segments', 'Pleth_n_segments', 'Resp_n_segments', 'rec_id_cum'])

    # copy across
    rec_link_with_rec_id_orig = df_expanded

    # add sequential rec_id
    rec_link_with_rec_id_orig['rec_id'] = [f'R{i}' for i in range(1, len(rec_link_with_rec_id_orig) + 1)]

    # - remove original rec ID
    rec_link = rec_link_with_rec_id_orig.drop(columns=['rec_id_orig'])
    
    # - save to file
    rec_link.to_csv(settings["paths"]["link_recording_all"], index=False)   
    print(f"     - Saved data for {len(rec_link)} recordings") 
    
    return rec_link_with_rec_id_orig


def compute_past_future_flags(group):

    # check whether visits are ordered chronologically
    if not all(earlier <= later for earlier, later in zip(group['arrival_time_dt'], group['arrival_time_dt'][1:])):
        raise ValueError("Group must be sorted by 'arrival_time_dt'")

    arrival_t = group['arrival_time_dt'].tolist()
    admitted = group['admitted'].values
    
    # Define visit outcomes
    visits = {
        'mi_visit': group['mi_diagnosis'].values,
        'cardiac_visit': group['cardiac_diagnosis'].values,
        'cv_visit': group['cardiovascular_diagnosis'].values,
        'any_visit': np.ones(len(group), dtype=bool)
    }
    
    # Also define hospitalization variants: need to AND with admitted
    hosp = {
        'cardiac_hosp': (visits['cardiac_visit'], admitted),
        'cv_hosp': (visits['cv_visit'], admitted),
        'hosp': (admitted,),  # just admitted flag for any hosp
    }
    
    # Initialize empty lists for all flags
    results = {}
    for key in list(visits.keys()) + list(hosp.keys()):
        results[key + '_after'] = []
        results[key + '_before'] = []
        results['time_to_' + key] = []
    
    # Loop through visits for each patient
    for i in range(len(group)):
        future = slice(i+1, None)
        past = slice(None, i)

        future_times = arrival_t[future]
        now_time = arrival_t[i]
        time_deltas = [(ft - now_time).days for ft in future_times]
        time_deltas = np.array(time_deltas)
                
        # Visits
        for key, mask in visits.items():
            past_mask = mask[past]
            future_mask = mask[future]

            results[key + '_before'].append(past_mask.any())
            results[key + '_after'].append(future_mask.any())

            # Time to event
            if future_mask.any():
                results['time_to_' + key].append(time_deltas[future_mask].min())
            else:
                results['time_to_' + key].append(np.nan)
        
        # Hospitalizations
        for key, masks in hosp.items():
            # Combine conditions using logical AND
            past_mask = np.logical_and.reduce([m[past] for m in masks])
            future_mask = np.logical_and.reduce([m[future] for m in masks])

            results[key + '_before'].append(past_mask.any())
            results[key + '_after'].append(future_mask.any())

            if future_mask.any():
                results['time_to_' + key].append(time_deltas[future_mask].min())
            else:
                results['time_to_' + key].append(np.nan)
    
    # Assign back to DataFrame
    for col, values in results.items():
        group[col] = values

    return group


def create_var_info():

    # Dictionary of variable descriptions
    var_descriptions = {
        'Visits': {'descrip': 'Number of ED visits per patient', 'type': 'num', 'overall_cat': 'pt_chars_baseline'},
        'Age': {'descrip': 'Age (years)', 'type': 'num', 'overall_cat': 'pt_chars_baseline'},
        'Gender': {'descrip': 'Gender', 'type': 'cat', 'overall_cat': 'pt_chars_baseline', 'categories': {'F': 'female', 'M': 'male', 'U': 'unknown'}},
        'Ethnicity': {'descrip': 'Ethnicity', 'type': 'cat', 'overall_cat': 'pt_chars_baseline', 'categories': {'nan': 'Unknown', 'Declines to State': 'Unknown'}},
        'Race': {'descrip': 'Race', 'type': 'cat', 'overall_cat': 'pt_chars_baseline', 'categories': {'nan': 'Unknown', 'Declines to State': 'Unknown'}},
    }

    # Sort the items based on the 'overall_cat' value
    var_descriptions = dict(sorted(var_descriptions.items(), key=lambda x: x[1]['overall_cat']))

    # overall cat descriptions
    overall_cat_descriptions = {
        'pt_chars_baseline': 'Patient characteristics at baseline',
        'monitoring': 'Monitoring',
        'follow_up': 'Follow-up',
    }

    var_info = {
        "num_vars": [key for key, value in var_descriptions.items() if value['type'] == 'num'],   # numerical variables
        "cat_vars": [key for key, value in var_descriptions.items() if value['type'] == 'cat'],   # categorical variables
        "var_descriptions": var_descriptions,
        "overall_cat_descriptions": overall_cat_descriptions,
    }

    return var_info


def reformat_variables(df, up):

    # Numerical Variables (specified in up)
    # replace commas with decimal points, and convert to float, in each numerical variable
    for var in up['num_vars']:

        if var not in df.columns:
            continue  # Skip to the next iteration if the column doesn't exist in the DataFrame
    
        # If the column is of type object (string), attempt to replace commas with decimal points
        if df[var].dtype == 'object':
            # Replace commas with periods in string columns
            df[var] = df[var].str.replace(',', '.', regex=False)
            # Try to convert the column to numeric, forcing errors to NaN (non-convertible entries)
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # Numerical variables (all)
    for col in df.columns:
        try:
            # Replace commas with dots and try to convert all values
            df[col] = df[col].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x)

            # Clean whitespace and replace empty strings with NaN
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = df[col].replace('', pd.NA)  # Replace empty strings with NaN

            # convert to numerical
            df[col] = pd.to_numeric(df[col], errors='raise')  # Raise error if any value can't be converted
        except:
            pass


    # Categorical
    # replace with intuitive strings
    for var in up['cat_vars']:

        if var not in df.columns:
            continue  # Skip to the next iteration if the column doesn't exist in the DataFrame
        
        # Extract the category mapping for 'fav_food'
        category_mapping = up['var_descriptions'][var]['categories']

        # Replace the categories in the 'fav_food' column with the new names
        df[var] = df[var].astype(str).replace(category_mapping)

    # String
    # - remove new line characters
    df = df.apply(lambda col: col.str.replace('\n', '', regex=False) if col.dtype == 'object' else col)   # Apply str.replace to each column to remove \n from string values

    # Days until next visit
    df["Days_until_next_visit"] = df["Hours_to_next_visit"]/24

    # Arrival time
    df['arrival_time_dt'] = df['Arrival_time'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')
    )

    # additional dataset-specific reformatting
    df = df.rename(columns={
        "MRN": "subj_id",
        "Visits": "no_visits",
        "Race": "race",
        "Ethnicity": "ethnicity",
        "Gender": "gender",
        "CSN": "visit_id",
        "Age": "age",
        "Triage_Temp": "temp",
        "Triage_HR": "heart_rate",
        "Triage_RR": "resp_rate",
        "Triage_SpO2": "spo2",
        "Triage_SBP": "sbp",
        "Triage_DBP": "dbp",
        "Hosp_LOS": "hosp_los",
        })
    gender_map = {'female': 1, 'male': 2}
    df['gender'] = df['gender'].map(gender_map)
    df['gender'] = df['gender'].astype('Int64')
    df = df.drop(columns=['Payor_class'])

    return df


def merge_df_waves_into_df(df_waves, df):

    # Pivot df_waves so that each type becomes a column with Duration values
    durations = df_waves.pivot_table(index='visit_id', columns='Type', values='Duration', fill_value=0)

    # Rename columns to match your desired output names
    durations = durations.rename(columns={
        'II': 'II_duration',
        'Pleth': 'Pleth_duration',
        'Resp': 'Resp_duration'
    })

    # Pivot df_waves so that each type becomes a column with Duration values
    n_segments = df_waves.pivot_table(index='visit_id', columns='Type', values='Segments', fill_value=0)

    # Rename columns to match your desired output names
    n_segments = n_segments.rename(columns={
        'II': 'II_n_segments',
        'Pleth': 'Pleth_n_segments',
        'Resp': 'Resp_n_segments'
    })

    # Merge with your main df (assuming 'visit_id' is the join key)
    df = df.merge(durations, on='visit_id', how='left')
    df = df.merge(n_segments, on='visit_id', how='left')

    # If any durations are still missing (e.g., visit_ids with no entries at all), fill with 0
    df[['II_duration', 'Pleth_duration', 'Resp_duration', 'II_n_segments', 'Pleth_n_segments', 'Resp_n_segments']] = df[['II_duration', 'Pleth_duration', 'Resp_duration', 'II_n_segments', 'Pleth_n_segments', 'Resp_n_segments']].fillna(0)

    # convert durations from secs to mins
    for col in ['II_duration', 'Pleth_duration', 'Resp_duration']:
        if col in df.columns:
            df[col] = df[col] / 60
    
    # convert n_segments to int
    for col in ['II_n_segments', 'Pleth_n_segments', 'Resp_n_segments']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df  


def identify_waveform_recording_file_paths(rec_link_with_rec_id_orig, settings):
    """
    Create file paths for waveform recordings and save them to a CSV.

    Parameters:
        settings (dict): Dictionary containing paths.

    Returns:
        ...
    """

    print("   - Creating list of paths for recording files")

    # recording link is passed as an input
    rec_link = rec_link_with_rec_id_orig
    
    # load encounter variables
    enc_vars = pd.read_csv(settings["paths"]["variables_encounter_all"])
    enc_vars = enc_vars[["enc_id", "visit_id"]]

    # merge
    rec_link = rec_link.merge(enc_vars, on='enc_id', how='left')

    # specify root folder
    filepaths_root = Path(settings["paths"]["dataset_root_raw_folder"]) / "waveforms"

    # obtain list of paths
    signal_filepaths = []
    for _, row in rec_link.iterrows():
        # - extract info on this recording
        rec_id_orig = str(row['rec_id_orig'])
        rec_id = str(row['rec_id'])
        visit_id = str(row["visit_id"])
        if row["II_available"]:
            avail_txt = 'II'
        elif row["Pleth_available"]:
            avail_txt = 'Pleth'
        elif row["Resp_available"]:
            avail_txt = 'Resp'
        else:
            raise Exception
        # - create filename
        filename = visit_id + '_' + rec_id_orig
        # - create filepath
        curr_path = str( Path(visit_id[-3:]) / visit_id / avail_txt / filename )
        # - add details to signal filepaths
        signal_filepaths.append({
            'rec_id': rec_id, 
            'filepath': curr_path, 
            'filetype': 'WFDB',
        })

    # store paths to file
    df = pd.DataFrame(signal_filepaths)
    df.to_csv(settings["paths"]["recording_filepaths_all"], index=False)

    # store root path to file
    with open(settings["paths"]["recording_filepaths_root_all"], 'w') as f:
        f.write(str(filepaths_root))

    return