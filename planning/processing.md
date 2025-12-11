# `longitudinal-ECG-analysis` Processing Pipeline

This flowchart summarises the `longitudinal-ECG-analysis` processing pipeline:

```mermaid
flowchart TD
    Step1["**Generate dataset settings** <br> _Generate generic dataset settings_ <br> **gen_dataset_settings.py**"] --> Step2
    Step2["**Curate entire dataset** <br> _Extract variables and metadata from entire dataset_ <br> **curate_entire_dataset.py**"] --> Step3
    Step2 --> Step2b
    Step2b["**Curate specific dataset** <br> _Dataset-specific curation_ <br> **curate_dataset_<...>.py**"] --> Step2
    Step3["**Generate analysis settings** <br> _Generate analysis-specific settings_ <br> **gen_analysis_settings.py**"] --> Step4
    Step4["**Curate analysis-specific dataset** <br> _Extract variables and metadata for a specific analysis_ <br> **curate_analysis_dataset.py**"] --> Step5
    Step5["**Derive features** <br> _Derive features from those signals required for the specific analysis_ <br> **derive_signal_features.py**"] --> Step6
    Step6["**Compile for stats** <br> _Aggregate variables (signal-derived features and other variables) at the encounter level_ <br> **compile_for_stats.py**"] --> Step7
    Step7["**Statistical analysis** <br> _Perform statistical analysis_ <br> **stats_analysis.py**"]
```

The entire pipeline can be run using **run_demo.py**

Further details of individual modules (corresponding to individual steps in the pipeline) are provided in [modules.md](./modules.md)