.. _datasets:

Datasets
========

The package integrates with two publicly available datasets containing ECG signals and health outcomes: the  :ref:`\MUSIC Dataset <music-dataset>`, and the :ref:`\MC-MED Dataset <mc-med-dataset>`. :ref:`\MC-MED Dataset <mc-med-dataset>`.

In addition, bespoke datasets can be analysed by creating a bespoke `dataset_curator` script.

.. _music-dataset:

MUSIC
-----

The `MUSIC (Sudden Cardiac Death in Chronic Heart Failure) dataset <https://doi.org/10.13026/fa8p-he52>`_ contains data from chronic heart failure patients attending heart failure clinics, including 24-hour Holter ECG recordings, additional clinical measurements such as left-ventricular ejection fraction (LVEF), and long-term outcomes. The original publication describing the dataset is `Martin et al. <https://www.cinc.org/archives/2024/pdf/CinC2024-355.pdf>`_.

Participant Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Considering those patients with Holter ECG data available, the following table shows the numbers of patients found to meet various criteria:

.. list-table:: The number of MUSIC patients with Holter ECG data available who met certain criteria. Bold indicates >= 100 per group.
   :widths: 20 40 40
   :header-rows: 1

   * - Outcome
     - No. with ECG available
     - No. with prior MI and ECG available
   * - Any
     - 936
     - 397
   * - Any-cause Death: Yes, No
     - **253, 683**
     - **141, 256**
   * - Sudden cardiac death: Yes, survived
     - 88, 683
     - 53, 256
   * - Pump-failure death: Yes, survived
     - **108, 683**
     - 58, 256
   * - Non-cardiac death: Yes, survived
     - 57, 683
     - 30, 256

.. _mc-med-dataset:

MC-MED
------

The `MC-MED (Multimodal Clinical Monitoring in the Emergency Department) dataset <https://doi.org/10.13026/jz99-4j81>`_ contains physiological signals from patients soon after arrival at the Emergency Department (ED), alongside short-term health outcomes. The original publication describing the dataset is `Kansal et al. <https://doi.org/10.1038/s41597-025-05419-5>`_.

Participant Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The entire dataset contains data from 118,385 ED visits by 70,545 patients. According to our analysis, physiological signals were recorded in 83,590 visits by 53,109 patients.

Considering for instance those patients who had a primary diagnosis of a myocardial infarction (an MI), the following table shows the numbers of patients found to meet various criteria:

.. list-table:: The number of MC-MED patients with a primary diagnosis of MI and ECG signals available who met certain criteria. Bold indicates >= 100 per group. NB: Some patients appear in multiple groups due to multiple visits.
   :widths: 20 80
   :header-rows: 1

   * - Outcome
     - No. patients with MI diagnosis and ECG signal available
   * - Any (all were admitted)
     - **542**
   * - Any-cause Death in Hospital: Yes, No
     - 11, 531
   * - Any-cause Rehospitalisation: Yes, No
     - **214, 352**
   * - CV Rehospitalisation: Yes, No
     - 88, 484
   * - Cardiac Rehospitalisation: Yes, No
     - 81, 491

