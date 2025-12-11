This file summarises the modules contained with `longitudinal-ECG-analysis`. It provides details of the inputs to each module, and the actions the module takes. A gentle introduction to the processing pipeline is provided in [processing.md](./processing.md).

---

# gen_dataset_settings

**inputs**
- `dataset_name`[^1]
- `dataset_root_raw_folder`[^2]
- `dataset_root_processing_folder`[^3]

**actions**
- create folders if they don't already exist:
  - `dataset_root_processing_folder`[^3]
  - `entire_dataset_proc_folder`[^4] (a subfolder)
  - `derived_features_proc_folder`[^5] (a subsubfolder) 
- create `dataset_settings.json`[^6] in `dataset_root_processing_folder`
- specify paths for standardised files for the entire dataset:
  - `link_encounter_all.csv`[^10]
  - `link_recording_all.csv`[^11]
  - `link_segment_all.csv`[^12]
  - `recording_filepaths_all.csv`[^16]
  - `recording_filepaths_root_all.txt`[^17]
  - `variables_subject_all.csv`[^13]
  - `variables_encounter_all.csv`[^14]
  - `variables_segment_all.csv`[^15]

---

# curate_entire_dataset

**inputs**
- `dataset_root_processing_folder`[^3]

**actions**
- load settings from `dataset_settings.json`[^6]
- extract variables and metadata from raw dataset files using dataset-specific script.
- convert to standardised format (using dataset-specific script), and write to standardised files in `entire_dataset_proc_folder`[^4]:
  - `link_encounter_all.csv`[^10]
  - `link_recording_all.csv`[^11]
  - `link_segment_all.csv`[^12]
  - `recording_filepaths_all.csv`[^16]
  - `recording_filepaths_root_all.txt`[^17]
  - `variables_subject_all.csv`[^13]
  - `variables_encounter_all.csv`[^14]
  - `variables_segment_all.csv`[^15]
- add in standardised metrics which can be derived from those available in the dataset to:
  - `variables_subject_all.csv`[^13]
  - `variables_encounter_all.csv`[^14]

---

# gen_analysis_settings

**inputs**
- `analysis_name`[^7]
- `dataset_root_processing_folder`[^3]
- an analysis specific settings file: `<analysis_name>_specific_settings.json`[^19], within the `dataset_root_processing_folder`[^3]

**actions**
- import custom analysis-specific settings from `<analysis_name>_specific_settings.json`[^19]
- create analysis-specific processing folder[^8] if it doesn't already exist
- create `analysis_settings.json`[^9] in the analysis-specific processing folder[^8]

---

# curate_analysis_dataset

**inputs**
- analysis name[^7]
- `dataset_root_processing_folder`[^3]

**actions**
- identify encounters to be included in the analysis, and store these in `rel_encs.csv`[^18] in the analysis-specific processing folder[^8]
- extract reduced set of variables and metadata from entire curated dataset for these encounters, and store in new files in the analysis-specific processing folder[^8]:
  - `link_encounter_analysis.csv`
  - `link_recording_analysis.csv`
  - `link_segment_analysis.csv`
  - `recording_filepaths_analysis.csv`
  - `recording_filepaths_root_analysis.txt`
  - `variables_subject_analysis.csv`
  - `variables_encounter_analysis.csv`
  - `variables_segment_analysis.csv`
- write to standardised files in analysis-specific processing folder[^8]

---

# derive_signal_features

**inputs**
- analysis name[^7]
- `dataset_root_processing_folder`[^3]

**actions**
- derive features from signal files as requried for this analysis (if not done already)
- create a new subfolder for each recording within `derived_features_proc_folder`[^5].
- write features to standardised files (one per signal) in each recording's subfolder within `derived_features_proc_folder`[^5].
- aggregate features for each signal and each encounter.
- create a new subfolder for each encounter within `derived_features_proc_folder`[^5].
- write these aggregated features to a standardised file (one per signal) in each encounter's subfolder within `derived_features_proc_folder`[^5].
- create a single file - `agg_enc_features.csv`[^20] within `derived_features_proc_folder`[^5] - containing aggregated features at the encounter level.

---

# compile_for_stats

**inputs**
- analysis name[^7]
- `dataset_root_processing_folder`[^3]

**actions**
- load signal-derived features aggregated at the encounter level from `agg_enc_features.csv`[^20].
- load remaining encounter-parameters from `variables_encounter_analysis.csv`
- merge to create single table of encounter-level variables.
- save to `predictor_response_table.csv`[^21] within the analysis-specific processing folder[^8].

---

# stats_analysis

**inputs**
- analysis name[^7]
- `dataset_root_processing_folder`[^3]

**actions**
- load predictor-response table from `predictor_response_table.csv`[^21].
- create table of encounter characteristics, and save to `tableone.csv`[^22].

---

# Further details
[^1]: **`dataset_name`**: a short name for a dataset, without spaces (_e.g._ `music`)
[^2]: **`dataset_root_raw_folder`**: the root folder containing the raw data for a dataset (_e.g._ `C:\data\music_raw`)
[^3]: **`dataset_root_processing_folder`**: the root folder containing the processed data for a dataset (_e.g._ `C:\data\music_proc`)
[^4]: **`entire_dataset_proc_folder`**: the folder containing processing for the entire dataset (_e.g._ `C:\data\music_proc\entire_dataset`)
[^5]: **`derived_features_proc_folder`**: the folder containing derived features for the entire dataset (_e.g._ `C:\data\music_proc\entire_dataset\derived_features`)
[^6]: **dataset_settings.json**: the file containing settings for the entire dataset (_e.g._ `C:\data\music_proc\entire_dataset\dataset_settings.json`)
[^7]: **`analysis_name`**: a short name for the analysis, without spaces (_e.g._ `analysis1`)
[^8]: **analysis-specific processing folder**: the folder containing the processed data for a specific analysis (_e.g._ `C:\data\music_proc\analysis1`)
[^9]: **analysis_settings.json**: the file containing settings for the specific analysis (_e.g._ `C:\data\music_proc\analysis1\analysis_settings.json`)
[^10]: **`link_encounter_all.csv`**: a link between `encounter_id` and `subject_id`, also containing encounter start and end times.
[^11]: **`link_recording_all.csv`**: a link between `recording_id` and `encounter_id`, also containing recording start and end times, and booleans indicating whether or not the recording contains each signal.
[^12]: **`link_segment_all.csv`**: a link between `segment_id` and `recording_id`, also containing segment start and end times.
[^13]: **`variables_subject_all.csv`**: initial subject-level variables derived from the dataset.
[^14]: **`variables_encounter_all.csv`**: initial encounter-level variables derived from the dataset.
[^15]: **`variables_segment_all.csv`**: initial segment-level variables derived from the dataset.
[^16]: **`recording_filepaths_all.csv`**: a list of filepaths for each recording.
[^17]: **`recording_filepaths_root_all.txt`**: the root folder for the recording filepaths.
[^18]: **`rel_encs.csv`**: a list of encounters to be included in the analysis.
[^19]: **<analysis_name>_specific_settings.json**: custom analysis-specific settings.
[^20]: **`agg_enc_features.csv`**: aggregate features for each encounter.
[^21]: **`predictor_response_table.csv`**: a table of encounter-level variables which can be used as predictor and response variables.
[^22]: **`tableone.csv`**: a table of encounter characteristics.