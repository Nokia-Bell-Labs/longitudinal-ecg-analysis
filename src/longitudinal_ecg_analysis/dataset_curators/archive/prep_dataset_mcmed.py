"""
prep_dataset_mcmed.py

Prepares the MC-MED (Multimodal Clinical Monitoring in the Emergency Department) dataset for analysis.
"""
import os
import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

def prep_dataset_mcmed(settings):
    """Prepare the MC-MED (Multimodal Clinical Monitoring in the Emergency Department) dataset for analysis.
    The dataset is available at: https://doi.org/10.13026/jz99-4j81

    Args:
        settings (dict): Settings loaded from a settings file.
        
    Returns:
        None. Writes the prepared data to disk.
    """

    # Check whether dataset is as expected
    settings = check_mcmed_dataset_files(settings)

    # Add dataset-specific settings
    settings = add_dataset_specific_settings(settings)

    # Prepare file of subject characteristics
    filepath = Path(settings["paths"]["mcmed_subj_chars_csv"])
    if settings["redo_analysis"] or (not filepath.exists()):
        create_subj_chars_file(settings)

    # Prepare file of visit characteristics
    filepath = Path(settings["paths"]["mcmed_visit_chars_csv"])
    if settings["redo_analysis"] or (not filepath.exists()):
        create_visit_chars_file(settings)

    # Extract standardised clinical, outcome and dataset variables
    filepath = Path(settings["paths"]["standard-outcome-variables-csv"])
    if settings["redo_analysis"] or (not filepath.exists()):
        clinical_metrics = extract_clinical_outcome_dataset_variables(settings)

    # Prepare file of waveform details:
    filepath = Path(settings["paths"]["mcmed_waveform_details_csv"])
    if settings["redo_analysis"] or (not filepath.exists()):
        create_waveform_details_file(settings)    

    # Create list of paths for ECG recording files
    filepath = Path(settings["paths"]["all-signal-filepaths-csv"])
    if settings["redo_analysis"] or (not filepath.exists()):
        file_paths = identify_signal_file_paths(settings)

    return


def create_waveform_details_file(settings):

    print(f" - Extracting waveform details")
    df = extract_waveform_details(settings["paths"]["waveform_folder"])
    df.to_csv(settings["paths"]["mcmed_waveform_details_csv"], index=False)
    print(f" - Saved waveform details to file")

    return


def extract_waveform_details(waveforms_root):
    waveform_data = []

    csn_suffix_folders = sort_numerical_folder_names(waveforms_root)

    for i, csn_suffix in enumerate(csn_suffix_folders, start=1):

        suffix_path = os.path.join(waveforms_root, csn_suffix)
        if not os.path.isdir(suffix_path):
            continue
        
        print(f'Processing {i} of {len(csn_suffix_folders)}')

        full_csn_folders = sort_numerical_folder_names(suffix_path)

        for full_csn in full_csn_folders:
            csn_path = os.path.join(suffix_path, full_csn)
            if not os.path.isdir(csn_path):
                continue

            for waveform_type in ['II', 'Pleth', 'Resp']:
                waveform_path = os.path.join(csn_path, waveform_type)
                if not os.path.isdir(waveform_path):
                    continue
                
                waveform_files = get_unique_waveform_basenames(waveform_path)

                for fname in waveform_files:
                    file_path_no_ext = os.path.splitext(os.path.join(waveform_path, fname))[0]
                    try:
                        header = wfdb.rdheader(file_path_no_ext)

                        start_datetime = None
                        if header.base_datetime:
                            start_datetime = header.base_datetime
                        elif header.date and header.time:
                            start_datetime = datetime.strptime(f"{header.date} {header.time}", "%d-%b-%Y %H:%M:%S")

                        waveform_data.append({
                            "CSN": full_csn,
                            "filename": header.record_name,
                            "waveform_type": waveform_type,
                            "waveform_start_time_date": start_datetime,
                            "duration": header.sig_len,
                            "sampling freq": header.fs
                        })

                    except Exception as e:
                        print(f"Failed to read {file_path_no_ext}: {e}")

    return pd.DataFrame(waveform_data)


def sort_numerical_folder_names(root_folder):
    
    sorted_folders = sorted(
        [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))],
        key=lambda x: int(x)
    )
    
    return sorted_folders


def get_unique_waveform_basenames(waveform_path):

    # Get all filenames (excluding extensions), deduplicated
    base_names = set(
        os.path.splitext(f)[0]
        for f in os.listdir(waveform_path)
        if f.endswith(('.dat', '.hea'))
    )

    # Sort by the numeric segment at the end (after "_")
    sorted_base_names = sorted(
        base_names,
        key=lambda x: int(x.split('_')[-1])
    )

    return sorted_base_names


def check_mcmed_dataset_files(settings):
    """
    Check the presence of required files and directories for the MC-MED dataset.

    This function verifies that the expected input files and folders exist in the
    directory specified by `settings["paths"]["input_dir"]`. It checks for:

    - A CSV file named 'visits.csv'
    
    These are essential for processing the dataset. If any of the required
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
    print(" - Checking dataset files")

    # Raw data

    # - Generate paths of raw data files
    settings["paths"]["visits_file_path"] = Path(settings["paths"]["input_dir"]) / "data/visits.csv"
    settings["paths"]["waveform_summary_file_path"] = Path(settings["paths"]["input_dir"]) / "data/waveform_summary.csv"
    settings["paths"]["waveform_folder"] = Path(settings["paths"]["input_dir"]) / "data/waveforms"
    
    # - Check whether required files and folders exist
    items_to_check = [
        settings["paths"]["visits_file_path"],
        settings["paths"]["waveform_summary_file_path"],
        settings["paths"]["waveform_folder"],
    ]
    for item in items_to_check:
        if not item.exists():
            raise FileNotFoundError(f'File not found at {item}')
    
    # - Generate paths of intermediate processing data files
    settings["paths"]["mcmed_subj_chars_csv"] = Path(settings["paths"]["intermediate_processing_dir"]) / "subj_chars.csv"
    settings["paths"]["mcmed_visit_chars_csv"] = Path(settings["paths"]["intermediate_processing_dir"]) / "visit_chars.csv"
    settings["paths"]["mcmed_waveform_details_csv"] = Path(settings["paths"]["intermediate_processing_dir"]) / "waveform_details.csv"
    
    return settings


def add_dataset_specific_settings(settings):

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
        'pt_chars_baseline': 'Patient characteristics',
        'monitoring': 'Monitoring',
        'follow_up': 'Follow-up',
    }

    settings["dataset_specific"] = {
        "num_vars": [key for key, value in var_descriptions.items() if value['type'] == 'num'],   # numerical variables
        "cat_vars": [key for key, value in var_descriptions.items() if value['type'] == 'cat'],   # categorical variables
        "var_descriptions": var_descriptions,
        "overall_cat_descriptions": overall_cat_descriptions,
        "admission_conditions": ['Ward', 'ICU', 'Observation'],
    }

    return settings


def create_subj_chars_file(settings):

    # Read the CSV into a DataFrame
    print(' - Loading raw data')
    df = load_dataset_from_csv(settings["paths"]["visits_file_path"])

    # remove unwanted variables
    df = df.drop(["CSN", "Visit_no", "Means_of_arrival", "Triage_Temp", "Triage_HR", "Triage_RR",
       "Triage_SpO2", "Triage_SBP", "Triage_DBP", "Triage_acuity", "CC",
       "ED_dispo", "Hours_to_next_visit", "Dispo_class_next_visit", "ED_LOS",
       "Hosp_LOS", "DC_dispo", "Payor_class", "Admit_service", "Dx_ICD9",
       "Dx_ICD10", "Dx_name", "Arrival_time", "Roomed_time", "Dispo_time",
       "Admit_time", "Departure_time", "Age"], axis=1)

    # Re-format variables
    print(' - Reformatting variables')
    df = reformat_variables(df, False, settings)

    # Save the dataframe
    print(' - Saving subj chars csv')
    df.to_csv(settings["paths"]["mcmed_subj_chars_csv"], index=False)

    return


def create_visit_chars_file(settings):

    # Read the CSV into a DataFrame
    print(' - Loading raw data')
    df = pd.read_csv(settings["paths"]["visits_file_path"], dtype={'Arrival_time': str})

    # Re-format variables
    print(' - Reformatting variables')
    df = reformat_variables(df, True, settings)

    # Derive additional variables
    print(' - Deriving additional variables')
    df = der_add_vars(df, settings)

    # remove unwanted variables
    df = df.drop(["Visits", "Gender", "Race", "Ethnicity", "Payor_class", "Dx_ICD9", "Dx_ICD10", "Roomed_time", "Dispo_time", "Admit_time", "Departure_time", "Hours_to_next_visit", "Dispo_class_next_visit", "ED_LOS", "Hosp_LOS", "Dx_name", "Arrival_time"], axis=1)

    # Read the waveform summary CSV into a DataFrame
    print(' - Loading raw data from Waveform Summary CSV')
    df_waves = pd.read_csv(settings["paths"]["waveform_summary_file_path"])
    
    # Merge df_waves into df
    df = merge_df_waves_into_df(df_waves, df)
    
    # derive waveform variables
    # - 'recorded' columns
    df['II_recorded'] = df['II_duration']>0
    df['Pleth_recorded'] = df['Pleth_duration']>0
    df['Resp_recorded'] = df['Resp_duration']>0
    df['Any_sig_recorded'] = (df['II_recorded']==1) | (df['Pleth_recorded']==1) | (df['Resp_recorded']==1)

    # Save the dataframe
    print(' - Saving dataframe')
    df.to_csv(settings["paths"]["mcmed_visit_chars_csv"], index=False)

    return df


def merge_df_waves_into_df(df_waves, df):

    # Pivot df_waves so that each type becomes a column with Duration values
    durations = df_waves.pivot_table(index='CSN', columns='Type', values='Duration', fill_value=0)

    # Optional: Rename columns to match your desired output names
    durations = durations.rename(columns={
        'II': 'II_duration',
        'Pleth': 'Pleth_duration',
        'Resp': 'Resp_duration'
    })

    # Merge with your main df (assuming 'CSN' is the join key)
    df = df.merge(durations, on='CSN', how='left')

    # If any durations are still missing (e.g., CSNs with no entries at all), fill with 0
    df[['II_duration', 'Pleth_duration', 'Resp_duration']] = df[['II_duration', 'Pleth_duration', 'Resp_duration']].fillna(0)

    # convert durations from secs to mins
    for col in ['II_duration', 'Pleth_duration', 'Resp_duration']:
        if col in df.columns:
            df[col] = df[col] / 60

    return df    


def load_dataset_from_csv(filepath):

    df = pd.read_csv(filepath)

    # take the first row per subject
    df_first_row = df.groupby('MRN').first().reset_index()

    return df_first_row


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


# Step 3: Define functions to compute "any future visit" conditions
def compute_past_future_flags_old(group):
    # Get values for current patient
    visit_nos = group['Visit_no'].values
    cardiac_visit = group['cardiac_diagnosis'].values
    cv_visit = group['cardiovascular_diagnosis'].values
    mi_visit = group['mi_diagnosis'].values
    admitted = group['admitted'].values
    arrival_t = group['arrival_time_dt'].values

    # Initialize result lists
    mi_visit_after = []
    cardiac_visit_after = []
    cv_visit_after = []
    cardiac_rehosp_after = []
    cv_rehosp_after = []
    any_rehosp_after = []
    mi_visit_before = []
    cardiac_visit_before = []
    cv_visit_before = []
    cardiac_hosp_before = []
    cv_hosp_before = []
    any_hosp_before = []
    
    # For each visit, look ahead in the same patient’s visits
    for i in range(len(group)):
        future = slice(i+1, None)
        past = slice(None, i)
        
        # Future masks
        mi_visit_after.append(mi_visit[future].any())
        cardiac_visit_after.append(cardiac_visit[future].any())
        cv_visit_after.append(cv_visit[future].any())
        cardiac_rehosp_after.append((cardiac_visit[future] & admitted[future]).any())
        cv_rehosp_after.append((cv_visit[future] & admitted[future]).any())
        any_rehosp_after.append(admitted[future].any())
    
        # Past masks
        mi_visit_before.append(mi_visit[past].any())
        cardiac_visit_before.append(cardiac_visit[past].any())
        cv_visit_before.append(cv_visit[past].any())
        cardiac_hosp_before.append((cardiac_visit[past] & admitted[past]).any())
        cv_hosp_before.append((cv_visit[past] & admitted[past]).any())
        any_hosp_before.append(admitted[past].any())
    
    group['mi_visit_after'] = mi_visit_after
    group['cardiac_visit_after'] = cardiac_visit_after
    group['cv_visit_after'] = cv_visit_after
    group['cardiac_rehosp_after'] = cardiac_rehosp_after
    group['cv_rehosp_after'] = cv_rehosp_after
    group['rehosp_after'] = any_rehosp_after
    group['mi_visit_before'] = mi_visit_before
    group['cardiac_visit_before'] = cardiac_visit_before
    group['cv_visit_before'] = cv_visit_before
    group['cardiac_hosp_before'] = cardiac_hosp_before
    group['cv_hosp_before'] = cv_hosp_before
    group['hosp_before'] = any_hosp_before
    
    return group


def der_add_vars(df, settings):

    # fill in unknown for blank ICD codes
    df['Dx_ICD10'] = df['Dx_ICD10'].fillna('Unknown')
    
    # Cardiac diagnoses
    print("   - cardiac / cardiovascular diagnoses")
    cardiac_pattern = r'^I(?:1[1-9]|[2-5][0-9])'  # True if the ICD10 code includes any of the strings I11, I12, I13, ..., or I59. Otherwise False.
    df['cardiac_diagnosis'] = df['Dx_ICD10'].str.contains(cardiac_pattern, regex=True)
    cardiovascular_pattern = r'^I(?:1[1-9]|[2-9][0-9])|^I99'  # True if the ICD10 code includes any of the strings I11, I12, I13, ..., or I99. Otherwise False.
    df['cardiovascular_diagnosis'] = df['Dx_ICD10'].str.contains(cardiovascular_pattern, regex=True)
    mi_pattern = r'^I(?:21|22)'  # True if the ICD10 code includes any of the strings I21 or I22. Otherwise False.
    df['mi_diagnosis'] = df['Dx_ICD10'].str.contains(mi_pattern, regex=True)
    subs_mi_pattern = r'^I22'  # True if the ICD10 code includes the string I22. Otherwise False.
    df['subsequent_mi_diagnosis'] = df['Dx_ICD10'].str.contains(subs_mi_pattern, regex=True)

    # Create 'admitted' column
    print("   - admission")
    df['admitted'] = df['ED_dispo'].isin(settings["dataset_specific"]["admission_conditions"])

    # Multiple ED visits
    print("   - multiple ED visits")
    df['visited_before'] = df['Visit_no']>1
    df['visited_after'] = df['Visit_no']<df['Visits']

    # Death in hospital
    print("   - death in hospital")
    df['death_in_hosp'] = df['DC_dispo'].isin(['Expired', 'Expired. Organ Harvest Complete', 'Expired to be Readmitted as Donor'])

    # Future visits
    print("   - past and future visits and admissions")
    df = df.sort_values(['MRN', 'Visit_no'])
    df = df.groupby('MRN', group_keys=False).apply(compute_past_future_flags)

    return df




def reformat_variables(df, visit_chars_log, settings):

    # Numerical Variables (specified in settings)
    # replace commas with decimal points, and convert to float, in each numerical variable
    for var in settings["dataset_specific"]["num_vars"]:

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
    for var in settings["dataset_specific"]["cat_vars"]:

        if var not in df.columns:
            continue  # Skip to the next iteration if the column doesn't exist in the DataFrame
        
        # Extract the category mapping for 'fav_food'
        category_mapping = settings["dataset_specific"]["var_descriptions"][var]['categories']

        # Replace the categories in the 'fav_food' column with the new names
        df[var] = df[var].astype(str).replace(category_mapping)

    # String
    # - remove new line characters
    df = df.apply(lambda col: col.str.replace('\n', '', regex=False) if col.dtype == 'object' else col)   # Apply str.replace to each column to remove \n from string values

    if visit_chars_log:
        
        # Days until next visit
        df["Days_until_next_visit"] = df["Hours_to_next_visit"]/24

        # Arrival time
        df['arrival_time_dt'] = df['Arrival_time'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')
        )


    return df


def extract_clinical_outcome_dataset_variables(settings):
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
    #if Path(settings["paths"]["standard-clinical-metrics-csv"]).exists() & Path(settings["paths"]["standard-dataset-variables-csv"]).exists() & Path(settings["paths"]["standard-outcome-variables-csv"]).exists():
    #    return

    # load raw data
    print(" - Loading raw tabular data")
    df_subj = pd.read_csv(settings["paths"]["mcmed_subj_chars_csv"], na_values=[])
    df_visit = pd.read_csv(settings["paths"]["mcmed_visit_chars_csv"], na_values=[])
    df = pd.merge(df_subj, df_visit, on='MRN', how='inner')
    
    # Remove visits where there wasn't a signal recorded
    df = df[df['Any_sig_recorded']==1]

    print(" - Extracting routine clinical parameters")

    # setup clinical metrics dataframe
    clinical_metrics = pd.DataFrame()

    # re-format standardised variables
    clinical_metrics['subj_id'] = df["MRN"]
    clinical_metrics['visit_id'] = df["CSN"]
    clinical_metrics['visit_no'] = df["Visit_no"]
    clinical_metrics['age'] = pd.to_numeric(df['Age'], errors='coerce')
    clinical_metrics['gender'] = df['Gender']
    clinical_metrics['gender'] = clinical_metrics['gender'].replace('M',2)  # male as 2
    clinical_metrics['gender'] = clinical_metrics['gender'].replace('F',1)  # female as 1
    clinical_metrics['dbp'] = pd.to_numeric(df['Triage_DBP'], errors='coerce')
    clinical_metrics['sbp'] = pd.to_numeric(df['Triage_SBP'], errors='coerce')
    clinical_metrics['temp'] = pd.to_numeric(df['Triage_Temp'], errors='coerce')
    clinical_metrics['hr'] = pd.to_numeric(df['Triage_HR'], errors='coerce')
    clinical_metrics['respr'] = pd.to_numeric(df['Triage_RR'], errors='coerce')
    clinical_metrics['spo2'] = pd.to_numeric(df['Triage_SpO2'], errors='coerce')
    clinical_metrics['prior_mi'] = pd.to_numeric(df['mi_diagnosis'], errors='coerce')
    clinical_metrics['race'] = df['Race']
    clinical_metrics['ethnicity'] = df['Ethnicity']
    clinical_metrics['means_of_arrival'] = df['Means_of_arrival']
    clinical_metrics['triage_acuity'] = df['Triage_acuity']
    clinical_metrics['chief_complaint'] = df['CC']
    clinical_metrics['admit_service'] = df['Admit_service']
    clinical_metrics['arrival_time'] = df['arrival_time_dt']
    clinical_metrics['cardiac_diagnosis'] = df['cardiac_diagnosis']
    clinical_metrics['cardiovascular_diagnosis'] = df['cardiovascular_diagnosis']
    
    # Add in additional variables
    
    # save clinical metrics to file
    clinical_metrics.to_csv(settings["paths"]["standard-clinical-metrics-csv"], index=False)

    print(" - Extracting dataset variables")

    # obtain rel variables
    dataset_variables = pd.DataFrame()
    dataset_variables['subj_id'] = df['MRN']
    dataset_variables['visit_id'] = df["CSN"]
    
    # - Signals available
    dataset_variables['Any_sig_recorded'] = pd.to_numeric(df['Any_sig_recorded'], errors='coerce')
    dataset_variables['II_recorded'] = pd.to_numeric(df['II_recorded'], errors='coerce')
    dataset_variables['Pleth_recorded'] = pd.to_numeric(df['Pleth_recorded'], errors='coerce')
    dataset_variables['Resp_recorded'] = pd.to_numeric(df['Resp_recorded'], errors='coerce')
    dataset_variables['II_duration'] = pd.to_numeric(df['II_duration'], errors='coerce')
    dataset_variables['Pleth_duration'] = pd.to_numeric(df['Pleth_duration'], errors='coerce')
    dataset_variables['Resp_duration'] = pd.to_numeric(df['Resp_duration'], errors='coerce')
    
    # save dataset variables to file
    dataset_variables.to_csv(settings["paths"]["standard-dataset-variables-csv"], index=False)

    print(" - Extracting outcome variables")

    outcome_variables = pd.DataFrame()
    print(df.columns)
    outcome_variables['subj_id'] = df["MRN"]
    outcome_variables['visit_id'] = df["CSN"]
    outcome_variables['ed_dispo'] = df["ED_dispo"]
    outcome_variables['hosp_dispo'] = df["DC_dispo"]
    outcome_variables['days_until_next_visit'] = df["Days_until_next_visit"]
    outcome_variables['subsequent_mi_diagnosis'] = df["subsequent_mi_diagnosis"]
    outcome_variables['admitted'] = df["admitted"]
    outcome_variables['visited_before'] = df["visited_before"]
    outcome_variables['visited_after'] = df["visited_after"]
    outcome_variables['mi_visit_before'] = df["mi_visit_before"]
    outcome_variables['time_to_mi_visit'] = df['time_to_mi_visit']
    outcome_variables['cardiac_visit_after'] = df["cardiac_visit_after"]
    outcome_variables['time_to_cardiac_visit'] = df['time_to_cardiac_visit']
    outcome_variables['cv_visit_after'] = df["cv_visit_after"]
    outcome_variables['time_to_cv_visit'] = df['time_to_cv_visit']
    outcome_variables['cardiac_hosp_after'] = df['cardiac_hosp_after']
    outcome_variables['time_to_cardiac_hosp'] = df['time_to_cardiac_hosp']
    outcome_variables['cv_hosp_after'] = df['cv_hosp_after']
    outcome_variables['time_to_cv_hosp'] = df['time_to_cv_hosp']
    outcome_variables['hosp_after'] = df['hosp_after']
    outcome_variables['time_to_hosp'] = df['time_to_hosp']
    outcome_variables['death_in_hosp'] = df["death_in_hosp"]
    # save to CSV
    outcome_variables.to_csv(settings["paths"]["standard-outcome-variables-csv"], index=False)
    
    return


def identify_signal_file_paths(settings):
    """
    Create file paths for signals and save them to a CSV.

    Parameters:
        settings (dict): Dictionary containing paths, including 'mcmed_waveform_details_csv'.

    Returns:
        pd.DataFrame: DataFrame with columns 'csn', 'rec_no', and 'filepath'.
    """

    print(" - Creating list of paths for ECG recording files")

    # create text file containing the root folder for the signal files:
    print("   - Saving root folder")
    root_folder = settings["paths"]["waveform_folder"]  # path of root folder
    output_path = settings["paths"]["all-signal-filepaths-root-txt"]  # path of file in which to store root folder
    with open(output_path, "w") as f:
        f.write(str(root_folder))  # write the root folder path to the file

    # obtain waveform details
    print("   - Creating list of filepaths")
    waveform_details = pd.read_csv(settings["paths"]["mcmed_waveform_details_csv"])
    
    # obtain list of paths
    records = []
    for row_no in range(0, len(waveform_details)):
        curr_filename = waveform_details.filename[row_no]
        curr_csn = str(waveform_details.CSN[row_no])
        temp = str(waveform_details.CSN[row_no])
        curr_csn_prefix = str(temp[-3:])
        curr_rec_no = curr_filename.split("_")[-1]
        curr_waveform_type = str(waveform_details.waveform_type[row_no])
        curr_path = Path(curr_csn_prefix) / curr_csn / curr_waveform_type / curr_filename
        records.append({'visit_id': curr_csn, 'rec_no': curr_rec_no, 'waveform_type': curr_waveform_type, 'filepath': curr_path, 'filetype': 'WFDB'})

    # store to file
    print("   - Saving list to file")
    df = pd.DataFrame(records)
    df.to_csv(settings["paths"]["all-signal-filepaths-csv"], index=False)

    return df


def summarise_pt_chars(df):
    """
    Summarise patient characteristics.

    Parameters
    ----------
    df : dataframe containing details of subjects

    Returns
    -------
    none
    """

    print("\n~~~ Patient Characteristics ~~~")
    print(f"Number of unique subjects: {len(df)}")
    print(f"Age, mean +/- SD: {df['age'].mean():.1f} +/- {df['age'].std():.1f}")
    print(f"LVEF, median (IQR): {df['lvef'].median():.1f} ({df['lvef'].quantile(0.25):.1f} - {df['lvef'].quantile(0.75):.1f})")
    print(f"Number with Holter available: {sum(df['holter_log'])}")
    print(f"Follow-up period (years), median (quartiles): {df['follow_up'].median():.1f} ({df['follow_up'].quantile(0.25):.1f} - {df['follow_up'].quantile(0.75):.1f})")
    print(f"Number all-cause death, Yes, No: {sum(df['outcome']!=0)}, {sum(df['outcome']==0)}")
    print(f"Number SCD, Yes, Survived: {sum(df['outcome']==3)}, {sum(df['outcome']==0)}") # see https://physionet.org/content/music-sudden-cardiac-death/1.0.1/subject-info_codes.csv
    print(f"Number pump-failure death, Yes, Survived: {sum(df['outcome']>=6)}, {sum(df['outcome']==0)}")
    print(f"Number non-cardiac death, Yes, Survived: {sum(df['outcome']==1)}, {sum(df['outcome']==0)}")
    print(f"Number previous MI, Yes, No: {sum(df['prior_mi']==1)}, {sum(df['prior_mi']==0)}")
    print(f"Number sinusal rhythm, Yes, No: {sum(df['sinusal_rhythm']==1)}, {sum(df['sinusal_rhythm']==0)}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    return
