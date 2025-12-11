# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

"""
derive_signal_features.py

Derives signal features for a particular analysis.
"""
import sys
import pandas as pd
import numpy as np
import wfdb as wfdb
import pyedflib
import neurokit2 as nk
import os
from pathlib import Path
from longitudinal_ecg_analysis.utils import standardise_name, load_analysis_settings, function_start_end_print, standardise_sig_name, load_recording_filepath_root, identify_recs
from longitudinal_ecg_analysis.feature_extractors import prsa
import pickle
import json

def derive_signal_features_for_analysis(
    dataset_root_proc_folder: str,
    analysis_name: str
):
    """Derive features from signals for a particular analysis.

    Args:
        dataset_root_proc_folder (str): Path to folder containing the settings file (named "dataset_settings.json") and in which to save the curated dataset.
        analysis_name (str): Name of the analysis (e.g. attempt1).
        
    Returns:
        None - results are saved to disk
        
    """

    # Load analysis settings from file
    settings = load_analysis_settings(dataset_root_proc_folder, analysis_name)

    # Start-matter
    function_start_end_print()
    print(f"Deriving signal features from '{settings['dataset_name']}' dataset for `{settings['analysis_name']}` analysis")

    # Extract features from each recording
    extract_features_from_recordings(settings)  

    # Aggregate features at encounter level
    aggregate_features_at_encounter_level(settings)

    # End-matter
    print(f" - Finished deriving signal features from '{settings['dataset_name']}' dataset for `{settings['analysis_name']}` analysis")
    function_start_end_print()

    return


def extract_features_from_recordings(settings):

    # Identify recordings to be analysed
    recs = identify_recs(settings, 'analysis')

    print(f" - Extracting features from {len(recs)} recordings:")

    # load recording filepath root
    filepath_root = load_recording_filepath_root(settings, 'analysis')

    # Identify potential signals:
    rel_sigs = identify_potential_signals(settings)

    # cycle through each signal to see whether it's present in this recording
    for curr_sig_dataset, curr_sig_standard in rel_sigs.items():

        print(f"   - '{curr_sig_standard}' signal")

        # name of signal availability column
        available_col = f"{curr_sig_dataset}_available"

        # Cycle through each recording:
        for _, row in recs.iterrows():

            # construct filepath of recording
            rec_filepath = Path(filepath_root) / row['filepath']

            # if the signal is available in this recording, then move to feature extraction
            if row.get(available_col, False):

                print(f"     - recording {row['rec_id']}")

                # specify filepath in which to save features extracted from this signal and this recording
                rec_sig_features_filepath = _create_rec_sig_features_filepath(settings, row['rec_id'], curr_sig_standard)

                # skip if these features have already been derived
                if rec_sig_features_filepath.exists() & (not settings["redo_derive_features"]):
                    print(f"     - Features already derived from: rec {row['rec_id']}; signal {curr_sig_standard} (denoted {curr_sig_dataset} in dataset)")
                    continue

                print(f"     - Deriving features from: rec {row['rec_id']}; signal {curr_sig_standard} (denoted {curr_sig_dataset} in dataset)")

                # Extract features from signal:
                metrics = derive_metrics_from_rec_sig(rec_filepath, row['filetype'], curr_sig_dataset, curr_sig_standard, settings)

                # Store the metadata about these metrics
                metrics['rec_id'] = row['rec_id']
                metrics['sig'] = curr_sig_standard

                # Save this subject's ECG metrics to file
                print(f"       - Saving features to {rec_sig_features_filepath}")
                metrics.to_csv(rec_sig_features_filepath, index=False)
    
    return


def aggregate_features_at_encounter_level(settings):

    # Identify recordings to be analysed
    recs = identify_recs(settings, 'analysis')

    # load recording filepath root
    filepath_root = load_recording_filepath_root(settings, 'analysis')

    # Identify potential signals:
    rel_sigs = identify_potential_signals(settings)

    # Identify encounters:
    enc_ids = recs['enc_id'].unique().tolist()

    agg_feats = []

    # cycle through each signal to see whether it's present in this recording
    for curr_sig_dataset, curr_sig_standard in rel_sigs.items():

        print(f"   - Aggregating features from signal: {curr_sig_standard}")

        # name of signal availability column
        available_col = f"{curr_sig_dataset}_available"

        # Cycle through each encounter:
        for enc_id in enc_ids:

            curr_recs = recs[recs['enc_id']==enc_id]

            enc_feats = []
             
            print(f"     - Aggregating features from encounter: {enc_id}")

            # Cycle through each recording:
            for _, row in curr_recs.iterrows():
                
                # if the signal is available in this recording, then move to loading features
                if row.get(available_col, False):

                    # specify filepath from which to load features extracted from this signal and this recording
                    rec_sig_features_filepath = _create_rec_sig_features_filepath(settings, row['rec_id'], curr_sig_standard)

                    # load features
                    feats = pd.read_csv(rec_sig_features_filepath)
                    
                    # store features
                    enc_feats.append(feats)
                    
            # aggregate features for this signal and this encounter
            if enc_feats:
                enc_feats = pd.concat(enc_feats, ignore_index=True)
                enc_feats = enc_feats.drop(columns=['seg_start','seg_end'], errors='ignore')
                enc_feats = enc_feats.median(numeric_only=True).to_frame().T

                # add enc_id as the first column
                enc_feats.insert(0, 'enc_id', enc_id)

                # store the aggregated features for this encounter and this signal
                agg_feats.append(enc_feats)

            else:
                print(f'     - Skipping encounter {enc_id} as no relevant data (perhaps CHANGE in future)')

    # convert to df
    agg_feats = pd.concat(agg_feats, ignore_index=True)

    # save to csv
    agg_feats.to_csv(settings["paths"]["aggregate_encounter_features"], index=False)
                    
    return


def identify_potential_signals(settings):

    # load recording IDs to be analysed
    with open(settings["paths"]["signal_channel_key_analysis"], "r") as f:
        signal_channel_key = json.load(f)
    
    return signal_channel_key


def _create_rec_sig_features_filepath(settings, rec_id, curr_sig):

    # specify filepath in which to save features extracted from this signal and this recording
    rec_sig_features_filepath = Path(settings["paths"]["derived_features_proc_folder"]) / rec_id / (rec_id + '_' + curr_sig +'.csv')

    # make sure the required subfolder for this recording exists
    os.makedirs(os.path.dirname(rec_sig_features_filepath), exist_ok=True)                

    return rec_sig_features_filepath


def derive_metrics_from_rec_sig(rec_filepath, filetype, sigtype_dataset, sigtype_standard, settings):

    # derive overall_sigtype
    overall_sig_type = sigtype_standard[0:3]  # e.g. 'ecg'

    # load info on signal duration
    sig_len, fs, channel_idx, edf_reader = load_info_on_signal(rec_filepath, filetype, sigtype_dataset)
    
    # define segments
    segs = define_segments(sig_len, fs, settings)

    # cycle through segments
    all_feats = []
    for _, row in segs.iterrows():

        print(f"         - Segment from {row['seg_starts']} - {row['seg_ends']} samples")

        # load signal segment
        if filetype=='WFDB':
            edf_reader_or_filepath = rec_filepath
        elif filetype=='EDF':
            edf_reader_or_filepath = edf_reader
        sig = load_sig_seg(edf_reader_or_filepath, fs=fs, sampfrom=row['seg_starts'], sampto=row['seg_ends'], channel_idx=channel_idx, filetype=filetype)

        # Extract features from this sig seg
        if overall_sig_type == 'ecg':
            # - ECG signal:

            # pre-processing:
            rec_filename = os.path.basename(rec_filepath)
            rec_filename_without_ext = os.path.splitext(rec_filename)[0]
            filename = rec_filename_without_ext + '_' + str(row['seg_starts']) + '_' + str(row['seg_ends']) + '_' + sigtype_standard + '_preproc.pkl'
            preproc_filepath = Path(settings["paths"]["preprocessing_proc_folder"]) / filename
            perform_preprocessing_for_ecg(sig, preproc_filepath, settings)

            # derive features:
            filename = os.path.basename(rec_filepath)[0:-4] + '_' + str(row['seg_starts']) + '_' + str(row['seg_ends']) + '_' + sigtype_standard + '_feats.pkl'
            feats_filepath = Path(settings["paths"]["preprocessing_proc_folder"]) / filename
            feats = derive_features_from_ecg(sig, preproc_filepath, feats_filepath, settings)
        else:
            raise Exception(f"Havent defined function to derive features from signal type {overall_sig_type}")
            
        # Add prefix to feats and metadata
        feats = add_prefix_to_columns(feats, sigtype_standard)
        feats['seg_start'] = row['seg_starts']
        feats['seg_end'] = row['seg_ends']
            
        # store
        all_feats.append(feats)
        
    # Combine all 1-row DataFrames into one big DataFrame
    if len(all_feats)>0:
        final_df = pd.concat(all_feats, ignore_index=True)
    else:
        print("         - No segments identified due to short signal")
        final_df = pd.DataFrame()

    return final_df


def load_sig_seg(edf_reader_or_filepath, fs, sampfrom, sampto, channel_idx, filetype):

    if filetype == 'EDF':

        # load signal of interest
        n_samples = sampto-sampfrom+1
        signal_data = edf_reader_or_filepath.readSignal(channel_idx, start=sampfrom, n=n_samples)
        
        # Extract signal for this segment
        sig = Signal(values=signal_data, fs=fs)

    elif filetype == 'WFDB':

        # Load segment of this signal
        seg = wfdb.rdrecord(edf_reader_or_filepath, sampfrom=sampfrom, sampto=sampto, channels=channel_idx)

        # Extract signal for this segment
        sig = Signal(values=seg.p_signal[:, 0], fs=seg.fs)

    else:

        raise Exception("Unknown signal filetype")

    return sig


def load_info_on_signal(rec_filepath, filetype, sigtype_dataset):
        
    if filetype == 'EDF':
        edf_reader = pyedflib.EdfReader(str(rec_filepath))
        signal_labels = edf_reader.getSignalLabels()
        rel_sigtype_dataset = identify_relevant_sigtype_dataset(sigtype_dataset, signal_labels)
        channel_idx = signal_labels.index(rel_sigtype_dataset)
        sig_len = edf_reader.getNSamples()[channel_idx]
        fs = edf_reader.getSampleFrequency(channel_idx)
        edf_reader_or_info = edf_reader
    
    elif filetype == 'WFDB':
        info = wfdb.rdheader(rec_filepath)
        sig_len = info.sig_len
        fs = info.fs
        rel_sigtype_dataset = identify_relevant_sigtype_dataset(sigtype_dataset, info.sig_name)
        channel_idx = [i for i, name in enumerate(info.sig_name) if name == rel_sigtype_dataset]
        edf_reader_or_info = info
    
    else:
        raise Exception("Unknown signal filetype")

    sig_len = int(sig_len)
    
    return sig_len, fs, channel_idx, edf_reader_or_info        


def identify_relevant_sigtype_dataset(init_sigtype_dataset, signal_labels):

    if init_sigtype_dataset in signal_labels:
        return init_sigtype_dataset
    else:
        trial_sigtype = init_sigtype_dataset.replace('_',' ')
        if trial_sigtype in signal_labels:
            return trial_sigtype
        trial_sigtype = init_sigtype_dataset.replace(' ','_')
        if trial_sigtype in signal_labels:
            return trial_sigtype
        raise Exception(f"No dataset sigtype found for {init_sigtype_dataset}")


def add_prefix_to_columns(metrics, prefix):
    """
    Add a prefix to column names in a DataFrame, excluding specified columns.

    This function adds the given prefix to all column names in the input
    DataFrame `metrics`, except for columns listed in `columns_to_skip`
    (which includes "EpochNo" and "subj_id").

    Parameters
    ----------
    metrics : pandas.DataFrame
        The DataFrame whose column names should be updated with a prefix.
    prefix : str
        The string to prepend to column names that are not excluded.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with updated column names.

    Notes
    -----
    Columns named "EpochNo" and "subj_id" will not be modified.
    """

    columns_to_skip = ["EpochNo", "subj_id"]

    prefix = prefix + '_'

    metrics_renamed = metrics.rename(columns={
        col: prefix + col for col in metrics.columns if col not in columns_to_skip
    })

    return metrics_renamed

def perform_preprocessing_for_ecg(sig, preproc_filepath, settings):
    """
    Pre-process ECG signal

    Parameters
    ----------
    sig : a signal (in the class Signal)
    settings : a dict of universal parameters

    Returns
    -------
    None - results are saved to disk
    """

    # See if this has been done already
    if preproc_filepath.exists() and (not settings["redo_derive_features"]):
        print(f"           - Pre-processing already done")
        return
    else: 
        # If not, then detect R-peaks
        print(f"           - Pre-processing signal of length {len(sig.v)} samples and duration {int(len(sig.v)/sig.fs)} secs")
        signals, info = nk.ecg_process(sig.v, sampling_rate=sig.fs)
        signals = signals[['ECG_R_Peaks', 'ECG_Rate']]  # retain only necessary variables
        print(f"           - Detected {sum(signals['ECG_R_Peaks'] == 1)} r-peaks")
        # Save to disk
        print("           - Saving pre-processing results")
        with open(preproc_filepath, 'wb') as f:
            pickle.dump({'signals': signals, 'info': info}, f)

    return



def derive_features_from_ecg(sig, preproc_filepath, feats_filepath, settings):
    """
    Derive metrics from an ECG signal

    Parameters
    ----------
    sig : a signal (in the class Signal)
    settings : a dict of universal parameters

    Returns
    -------
    metrics : a dataframe of metrics
    """

    # Skip if features have already been derived
    if feats_filepath.exists() and (not settings["redo_derive_features"]):
        # load saved feats
        print("           - Loading saved features")
        median_df = pd.read_pickle(feats_filepath)
        return median_df
        
    # Load pre-processing variables
    with open(preproc_filepath, 'rb') as f:
        preproc_data = pickle.load(f)
        signals = preproc_data['signals']
        info = preproc_data['info']
    
    # Segment signal into epochs
    epochs = nk.epochs_create(signals, sampling_rate=sig.fs, epochs_end=settings["parameters"]["epoch_duration"])
    print(f"           - Split into {len(epochs)} epochs, each of duration {settings['parameters']['epoch_duration']} seconds.")
    
    # Calculate Heart Rates
    print("           - Calculating heart rates")
    hr_summary = nk.ecg_intervalrelated(epochs)
    hr_summary['EpochNo'] = hr_summary['Label']
    hr_summary = hr_summary.drop(columns=['Label'])
    print(f"               (median HR: {hr_summary['ECG_Rate_Mean'].median():.1f} bpm)")
    
    # Calculate HRV metrics
    print("           - Calculating HRV metrics")
    hrv_results = []
    for label, epoch in epochs.items():
        try:
            # Compute HRV time-domain features for the epoch
            hrv = nk.hrv(epoch, sampling_rate=sig.fs, show=False)
            hrv["EpochNo"] = label
            hrv["Start_el"] = epoch["Index"][0]
            hrv["End_el"] = epoch["Index"].iloc[-1]
            hrv_results.append(hrv)
        except Exception as e:
            print(f"Skipping {label} due to error: {e}")
    
    hrv_df = pd.concat(hrv_results, ignore_index=True)

    # Merge HR and HRV metrics
    # - Merge with suffixes so you can identify overlapping columns
    metrics = pd.merge(hr_summary, hrv_df, on='EpochNo', suffixes=('_summary', ''))  # Keep hrv_df version clean
    # - Drop columns from hr_summary that were duplicated (those with '_summary' suffix)
    metrics = metrics[[col for col in metrics.columns if not col.endswith('_summary')]]

    # Calculate PRSA metrics
    print("           - Calculating PRSA metrics")
    prsa_metrics = prsa.extract_prsa_metrics(epochs, sig.fs, do_verbose=False)

    # Merge in PRSA metrics
    # - Merge with suffixes so you can identify overlapping columns
    metrics = pd.merge(metrics, prsa_metrics, on='EpochNo', suffixes=('', '_summary'))  # keep metrics clean
    # - Drop columns that were duplicated (those with '_summary' suffix)
    metrics = metrics[[col for col in metrics.columns if not col.endswith('_summary')]]
    
    # Compute median for those columns
    cols_to_median = metrics.columns.difference(['Start_el', 'End_el', 'EpochNo'])  # # Select all columns except these
    median_values = metrics[cols_to_median].median()

    # Convert the resulting Series to a DataFrame (one row)
    median_df = median_values.to_frame().T

    # Save to disk
    print("           - Saving features")
    median_df.to_pickle(feats_filepath)

    return median_df


def define_segments(sig_len, fs, settings):

    # - signal duration
    durn = sig_len/fs

    # - start and end samples of period to be analysed
    period_start_samp = int(settings["parameters"]["buffer_to_discard"]*fs)  # in samples
    max_len = int(settings["parameters"]["signal_duration_to_analyse"]*fs)  # in samples
    durn_samps = int(min(sig_len-period_start_samp, max_len))
    period_end_samp = int(period_start_samp + durn_samps)

    # - identify segments within this period
    seg_durn_samps = int(settings["parameters"]["window_duration"]*fs)   # in samps
    seg_delineation = np.arange(period_start_samp, period_end_samp + 1, seg_durn_samps)
    seg_starts = seg_delineation[:-1]
    seg_ends = seg_delineation[1:]

    # - convert to df
    segs = pd.DataFrame({
        'seg_starts': seg_starts,
        'seg_ends': seg_ends
    })

    print(f'       - Signal of duration {int(durn)} secs, split into {len(segs)} segments, each of duration {settings["parameters"]["window_duration"]} seconds and fs {fs} Hz.')

    return segs


class Signal:
    """
    Represents a time-series signal with sampling information.

    This class stores the signal values, the sampling frequency,
    and automatically generates a time vector aligned with the values.

    Attributes
    ----------
    v : array-like
        The signal values (e.g., amplitude samples).
    fs : float
        Sampling frequency in Hz (samples per second).
    t : numpy.ndarray
        Time vector corresponding to the signal values, computed as t = np.arange(len(v)) / fs.

    Methods
    -------
    _generate_time_vector()
        Generate a time vector based on the signal length and sampling frequency.
    """
    def __init__(self, values, fs):
        self.v = values
        self.fs = fs
        self.t = self._generate_time_vector()

    def _generate_time_vector(self):
        return np.arange(len(self.v)) / self.fs


if __name__ == "__main__":

    # Check whether expected number of inputs have been provided
    if len(sys.argv) != 3:
        print("Usage: python -m longitudinal_ecg_analysis.derive_signal_features <dataset_root_proc_folder> <analysis_name>")
        sys.exit(1)

    # parse inputs
    dataset_root_proc_folder = sys.argv[1]
    analysis_name = sys.argv[2]

    # standardise analysis name
    analysis_name = standardise_name(analysis_name)

    # call function to derive signal features
    derive_signal_features_for_analysis(dataset_root_proc_folder, analysis_name)
