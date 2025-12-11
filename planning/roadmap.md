# Developing `longitudinal-ECG-analysis`

This file lists planned steps in developing `longitudinal-ECG-analysis`.


---

## Compatability with datasets

1. Make compatible with MC-MED:
   - check dataset runs with new recording duration extraction (I have implemented recording duration for non-WFDB file type, but now doesn't seem to find one of the files)
   - develop an initial set of inclusion criteria (e.g. only MI patients, include recording duration as a new inclusion criterion)
   - make sure I've handled visits with no waveform recordings (I think this is done)
   - handle encounters with only some (and not all) of the signals (perhaps this will be done by adapting the inclusion criteria to state what signal(s) must be present in an encounter for it to included in the analysis)
   - handle multiple WFDB recording files per encounter, which may require concatenation (although they may not be contiguous)
   - derive outcome variables (using approaches from previous mcmed scripts)
   - add in: functionality for PPG signal processing
   - add in: functionality for Resp signal processing
   - add in: variable outcomes 

## Signal-derived features

2. PPG-derived features?
3. Foundation model features?

## Statistical analysis

1. Implement stats from Ramirez et al [^1]:
   - univariable hazard ratios using Cox Regression (Table 3)
   - multivariable cox regression to develop models using backward stepwise elimination (or consider alternatives such as LASSO).
   - illustrate separation between centile groups (e.g. quintiles) using kaplan-meier curves. Perhaps look into stats for this, e.g. perhaps Harrell's C-index.
   - AUROC for binary outcomes

## Tidying up

1. Update comments in files:
   - to ensure docs include descriptions of modules and functions
   - to ensure descriptions of functions are up to date

## Testing

1. Need to check that data processing is correct (e.g. check that information in predictor-response table is correct for a couple of subjects)
2. Would be very helpful to see whether we get similar results to those published previously (e.g. Ramirez et al on the MUSIC dataset; HH publication)


---

# Experiments

1. Comparing waveform characteristics between groups (e.g. those with and without CV disease; prior MI; heart failure; ) - MC-MED
2. Predicting short-term CV triage outcomes (i.e. whether or not admitted to hospital) - MC-MED
   - perhaps select those with a certain chief complaint which is indicative of a potential CV problem and where there is a mix of people admitted and discharged (e.g. shortness of breath, chest pain)
   - compare predictive power between using all clinical data, waveform-derived variables alone, and both together.
   - could you have an algorithm with high specificity to predict admissions based on waveform-derived variables alone?
3. Predicting long-term CV outcomes:
   - Rehospitalisation (HH)
   - Deaths (MUSIC)


---

# References

[^1]: **`Ramirez et al`**: 'Sudden cardiac death and pump failure death prediction in chronic heart failure by combining ECG and clinical markers in an integrated risk model', https://doi.org/10.1371/journal.pone.0186152