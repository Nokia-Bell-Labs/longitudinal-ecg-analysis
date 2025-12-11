# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
compile_for_stats.py

Compile data for statistical analysis.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from longitudinal_ecg_analysis.utils import standardise_name, load_analysis_settings, function_start_end_print

def do_compile_for_stats(dataset_root_proc_folder, analysis_name):

    # Load analysis settings from file
    settings = load_analysis_settings(dataset_root_proc_folder, analysis_name)

    # Start-matter
    function_start_end_print()
    print(f"Compiling data for statistical analysis in `{settings['analysis_name']}` analysis")

    # Load signal features at encounter level
    print(f" - Loading encounter-level data")
    agg_feats = pd.read_csv(settings["paths"]["aggregate_encounter_features"])
    
    # Load remaining encounter parameters
    enc_params = pd.read_csv(settings["paths"]["variables_encounter_analysis"])
    
    # Merge, joining on enc_id, with inner merge to get rid of additional encounters in agg_feats which aren't required for this analysis
    print(f" - Merging encounter-level data")
    enc_vars = pd.merge(enc_params, agg_feats, on="enc_id", how="inner")

    # Add in subject id
    link_enc = pd.read_csv(settings["paths"]["link_encounter_all"])  # contains enc_id and subj_id
    enc_vars = pd.merge(enc_vars, link_enc, on='enc_id', how='inner')

    # Move 'subj_id' to be the second column (after 'enc_id')
    cols = enc_vars.columns.tolist()
    cols_reordered = (['enc_id', 'subj_id'] + [col for col in cols if col not in ['enc_id', 'subj_id']])
    enc_vars = enc_vars[cols_reordered]

    # Save to CSV
    print(f" - Saving encounter-level data")
    enc_vars.to_csv(settings["paths"]["predictor_response_table"], index=False)

    # End-matter
    function_start_end_print()

    return


if __name__ == "__main__":

    # Check whether expected number of inputs have been provided
    if len(sys.argv) != 3:
        print(sys.argv)
        print("Usage: python -m longitudinal_ecg_analysis.compile_for_stats <dataset_root_proc_folder> <analysis_name>")
        sys.exit(1)

    # Parse inputs
    dataset_root_proc_folder = sys.argv[1]
    analysis_name = sys.argv[2]

    # Standardise analysis name
    analysis_name = standardise_name(analysis_name)

    # Call function to compile data for statistical analysis
    do_compile_for_stats(dataset_root_proc_folder, analysis_name)
