# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

import numpy as np
import pandas as pd

def calculate_prsa_features(rri_series, segment_size=4, threshold=0.05, do_verbose=True):
    """
    Calculate Phase Rectification Signal Averaging (PRSA) features from RR interval series.
    
    Parameters
    ----------
    rri_series : array-like
        Series of RR intervals in milliseconds.
    segment_size : int, optional
        Size of segments around anchors, default is4 beats.
    threshold : float, optional
        Threshold for excluding RR prolongations/shortenings (as percentage), default is 0.05 (5%).
    
    Returns
    -------
    dict
        Dictionary containing:
        - DC: Deceleration Capacity
        - AC: Acceleration Capacity
        - DC_PRSA: The PRSA signal for deceleration
        - AC_PRSA: The PRSA signal for acceleration
        - DC_anchors: Count of anchors used for DC
        - AC_anchors: Count of anchors used for AC
    """
    rri = np.array(rri_series)
    
    # Step 1: 
    percent_changes = np.diff(rri) / rri[:-1]

    # For DC: intervals longer than preceding, but not by more than threshold
    dc_anchors = np.where((percent_changes > 0) & 
                        (percent_changes <= threshold))[0] + 1

    if do_verbose:
        print(f"DC Anchors: {len(dc_anchors)}")
        print(f"{len(np.where(percent_changes > 0)[0]) - len(dc_anchors)} anchors filtered out by the threshold of {threshold * 100}%")

    # For AC: intervals shorter than preceding, but not by more than threshold
    ac_anchors = np.where((percent_changes < 0) & 
                        (percent_changes >= -threshold))[0] + 1

    if do_verbose:
        print(f"AC Anchors: {len(ac_anchors)}")
        print(f"{len(np.where(percent_changes < 0)[0]) - len(ac_anchors)} anchors filtered out by the threshold of {threshold * 100}%")

    # Define the half-window size
    half_size = segment_size // 2
    
    # Step 2-4: Define segments, phase rectification, and averaging
    # For DC
    dc_segments = []
    for anchor in dc_anchors:
        if anchor >= half_size and anchor < len(rri) - half_size:
            segment = rri[anchor - half_size:anchor + half_size]
            dc_segments.append(segment)
    
    # For AC
    ac_segments = []
    for anchor in ac_anchors:
        if anchor >= half_size and anchor < len(rri) - half_size:
            segment = rri[anchor - half_size:anchor + half_size]
            ac_segments.append(segment)
    
    # Convert to numpy arrays for efficiency
    if dc_segments:
        dc_segments = np.array(dc_segments)
        dc_prsa = np.mean(dc_segments, axis=0)
    else:
        dc_prsa = np.zeros(segment_size)
    
    if ac_segments:
        ac_segments = np.array(ac_segments)
        ac_prsa = np.mean(ac_segments, axis=0)
    else:
        ac_prsa = np.zeros(segment_size)
    
    # Step 5: Quantification of DC and AC
    # The central deflection is quantified using Haar wavelet analysis
    # DC = [X(0) + X(1) - X(-1) - X(-2)] / 4
    center = half_size
    
    if len(dc_prsa) >= segment_size and center + 1 < len(dc_prsa) and center - 2 >= 0:
        dc = (dc_prsa[center] + dc_prsa[center+1] - dc_prsa[center-1] - dc_prsa[center-2]) / 4
    else:
        dc = np.nan
        
    if len(ac_prsa) >= segment_size and center + 1 < len(ac_prsa) and center - 2 >= 0:
        ac = (ac_prsa[center] + ac_prsa[center+1] - ac_prsa[center-1] - ac_prsa[center-2]) / 4
    else:
        ac = np.nan
    
    return {
        'DC': dc,
        'AC': ac,
        'DC_PRSA': dc_prsa,
        'AC_PRSA': ac_prsa,
        'DC_anchors': len(dc_anchors),
        'AC_anchors': len(ac_anchors)
    }


def extract_prsa_metrics(epochs, sampling_rate, do_verbose=True):
    """
    Extract PRSA metrics from ECG epochs.
    
    Parameters
    ----------
    epochs : dict
        Dictionary of segmented ECG signals from neurokit2
    sampling_rate : float
        The sampling frequency of the ECG signal (Hz)
    do_verbose : bool
        Whether or not to print intermediate results
        
    Returns
    -------
    DataFrame
        DataFrame containing PRSA metrics for each epoch

    References
    ----------
    A. Bauer et al., 'Deceleration capacity of heart rate as a predictor of mortality after myocardial infarction: cohort study', The Lancet, vol. 367, no. 9523, pp. 1674–1681, May 2006, doi: 10.1016/S0140-6736(06)68735-7.
    A. Bauer et al., 'Phase-rectified signal averaging detects quasi-periodicities in non-stationary data', Physica A: Statistical Mechanics and its Applications, vol. 364, pp. 423–434, May 2006, doi: 10.1016/j.physa.2005.08.080.

    """
    prsa_results = []
    
    for label, epoch in epochs.items():
        try:
            # Get RR intervals from the epoch
            # Extract R-peaks and calculate RR intervals
            rpeaks = np.where(epoch["ECG_R_Peaks"] == 1)[0]
            if len(rpeaks) < 10:  # Need minimum number of R-peaks for PRSA calculation
                if do_verbose:
                    print(f"Skipping {label}: Too few R-peaks ({len(rpeaks)})")
                continue
                
            # Calculate RR intervals in ms
            rri = np.diff(rpeaks) / sampling_rate * 1000
            
            # Calculate PRSA metrics
            prsa_metrics = calculate_prsa_features(rri, segment_size=4, threshold=0.05, do_verbose=do_verbose)
            
            # Add epoch information
            prsa_metrics["EpochNo"] = label
            prsa_metrics["Start_el"] = epoch["Index"][0]
            prsa_metrics["End_el"] = epoch["Index"].iloc[-1]
            
            prsa_results.append(prsa_metrics)
        except Exception as e:
            if do_verbose:
                print(f"Error processing {label}: {e}")
    
    # Convert list of dictionaries to DataFrame
    if prsa_results:
        df = pd.DataFrame(prsa_results)
        # Remove array columns to make it compatible with CSV export
        if 'DC_PRSA' in df.columns:
            df = df.drop(columns=['DC_PRSA', 'AC_PRSA', 'DC_anchors', 'AC_anchors'])
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['EpochNo', 'DC', 'AC', 'Start_el', 'End_el'])
