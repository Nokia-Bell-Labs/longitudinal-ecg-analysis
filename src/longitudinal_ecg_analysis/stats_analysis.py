# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
stats_analysis.py

Perform statistical analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats # For statistical tests
from longitudinal_ecg_analysis.utils import standardise_name, load_analysis_settings, function_start_end_print, identify_predictor_response_variables
from tableone import TableOne

def do_stats_analysis(dataset_root_proc_folder, analysis_name):

    # Load analysis settings from file
    settings = load_analysis_settings(dataset_root_proc_folder, analysis_name)

    # Start-matter
    function_start_end_print()
    print(f"Performing statistical analysis for `{settings['analysis_name']}` analysis")

    # Load encounter-level variables
    print(f" - Loading encounter-level data")
    enc_vars = pd.read_csv(settings["paths"]["predictor_response_table"])

    # drop unwanted variablxes
    enc_vars = enc_vars.drop(columns=['subj_id', 'enc_id'], errors='ignore')

    # identify predictor and response variables
    vars = enc_vars.columns.to_list()
    categories = identify_predictor_response_variables(vars)
    predictor_vars = categories["predictors"]
    response_vars = categories["responses"]
    time_to_response_vars = categories["time_to_responses"]
    var_types = categories["variable_types"]
    
    # make tableone
    print(f" - Making table of encounter characteristics")
    make_tableone(enc_vars, var_types, settings)

    ## Statistical analysis
    print(f" - Performing analysis")
    do_univariate_analysis(enc_vars, response_vars, predictor_vars, settings)

    # End-matter
    function_start_end_print()

    return


# Updated function to handle numeric, binary, and categorical predictors with >2 levels
def do_univariate_analysis(enc_vars, response_vars, predictor_vars, settings):
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from scipy import stats

    print(f"   - Univariate analysis")

    # cycle through response vars
    for response_var in response_vars:

        print(f"     - Response var: {response_var}")
        
        # Identify outcome groups
        unique_outcome_groups = sorted(enc_vars[response_var].dropna().unique())

        # --- 2. Prepare for Results Storage ---
        results_rows = []

        # --- 3. Add row for number of patients in each outcome group ---
        group_counts = enc_vars[response_var].value_counts().sort_index()
        count_row = {'Predictor Variable': 'N'}
        for group_id in unique_outcome_groups:
            count_row[f'Group {group_id} value'] = int(group_counts.get(group_id, 0))
            count_row[f'Group {group_id} vs others p-value'] = '-'
            count_row[f'Group {group_id} vs others sig'] = '-'
        results_rows.append(count_row)

        # --- 4. Loop Through Predictor Variables and Outcome Groups ---
        for predictor in predictor_vars:
            row_data = {'Predictor Variable': predictor}
            predictor_series = enc_vars[predictor]
            predictor_non_na = predictor_series.dropna()
            is_binary = predictor_non_na.isin([0, 1]).all()
            is_numeric = np.issubdtype(predictor_non_na.dtype, np.number)
            is_categorical = predictor_series.dtype == 'object' or predictor_series.dtype.name == 'category'

            for group_id in unique_outcome_groups:
                data_in_group = enc_vars[enc_vars[response_var] == group_id][predictor].dropna()
                value_str = "N/A (No data)"
                if not data_in_group.empty:
                    if is_binary:
                        n_ones = (data_in_group == 1).sum()
                        perc_ones = 100 * n_ones / len(data_in_group)
                        value_str = f"{n_ones} ({perc_ones:.2f} %)"
                    elif is_numeric:
                        median_val = data_in_group.median()
                        Q1 = data_in_group.quantile(0.25)
                        Q3 = data_in_group.quantile(0.75)
                        iqr_val = Q3 - Q1
                        value_str = f"{median_val:.2f} ({iqr_val:.2f})"
                    else:
                        value_counts = data_in_group.value_counts()
                        value_str = "; ".join([f"{val}: {count}" for val, count in value_counts.items()])
                row_data[f'Group {group_id} value'] = value_str

                # Compare with other groups
                data_out_group = enc_vars[enc_vars[response_var] != group_id][predictor].dropna()
                p_value = np.nan

                if not data_in_group.empty and not data_out_group.empty:
                    try:
                        if is_binary or is_numeric:
                            # Mann–Whitney U for numeric and binary
                            stat, p_value = stats.mannwhitneyu(data_in_group, data_out_group, alternative='two-sided')
                        elif is_categorical:
                            # Chi-squared test on contingency table
                            table = pd.crosstab(enc_vars[predictor], enc_vars[response_var])
                            if table.shape[0] > 1 and table.shape[1] > 1:
                                _, p_value, _, _ = stats.chi2_contingency(table)
                            # could add kruskal-wallis for ordinal categories (e.g. Killip class)
                    except Exception as e:
                        print(f"Warning: Could not compute test for {predictor} (Group {group_id} vs others): {e}")
                        p_value = np.nan

                # Format p-value
                if pd.isna(p_value):
                    p_value_str = "N/A"
                    sig_str = "N/A"
                elif p_value < 0.001:
                    p_value_str = "<0.001"
                    sig_str = "1"
                else:
                    p_value_str = f"{p_value:.3f}"
                    sig_str = "1" if p_value < settings['parameters']['alpha'] else "0"

                row_data[f'Group {group_id} vs others p-value'] = p_value_str
                row_data[f'Group {group_id} vs others sig'] = sig_str

            results_rows.append(row_data)

        # --- 5. Create Final DataFrame and Save ---
        final_df = pd.DataFrame(results_rows)

        # Reorder columns
        ordered_columns = ['Predictor Variable']
        for group_id in unique_outcome_groups:
            ordered_columns.extend([
                f'Group {group_id} value',
                f'Group {group_id} vs others p-value',
                f'Group {group_id} vs others sig'
            ])
        final_df = final_df[ordered_columns]

        # Save
        output_path = Path(settings["paths"]["analysis_res_folder"]) / f"univariate_{response_var}.csv"
        final_df.to_csv(output_path, index=False)

    return


def make_tableone(enc_vars, var_types, settings):

    columns = list(enc_vars.columns)
    categorical_vars = [var for var, dtype in var_types.items() if dtype == "categorical"]
    numerical_vars = [var for var, dtype in var_types.items() if dtype == "numerical"]

    mytable = TableOne(enc_vars, columns=columns, categorical=categorical_vars, continuous=numerical_vars, dip_test=False, normal_test=False, tukey_test=False, show_histograms=False)

    mytable.to_csv(settings["paths"]["tableone"])
    
    return

if __name__ == "__main__":
    import sys

    # Check whether expected number of inputs have been provided
    if len(sys.argv) != 3:
        print(sys.argv)
        print("Usage: python -m longitudinal_ecg_analysis.stats_analysis <dataset_root_proc_folder> <analysis_name>")
        sys.exit(1)

    # Parse inputs
    dataset_root_proc_folder = sys.argv[1]
    analysis_name = sys.argv[2]

    # Standardise analysis name
    analysis_name = standardise_name(analysis_name)

    # Do statistical analysis
    do_stats_analysis(dataset_root_proc_folder, analysis_name)
