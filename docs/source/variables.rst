Variables
=========

**Longitudinal ECG Analysis** extracts standard variables from datasets, either by extracting variables recorded in routine clinical practice or studies (such as age and gender), or by deriving variables from ECG signals (such as heart rate).

.. _link-variables:

Link variables
--------------

The following link variables are used:

.. list-table:: Link variables
   :widths: 20 80
   :header-rows: 1

   * - Abbreviation
     - Details
   * - subj_id
     - Subject ID (a string)
   * - enc_id
     - Encounter ID (a string): one subject can have multiple encounters
   * - rec_id
     - Recording ID (a string): one encounter can have multiple recordings


.. _clinical-variables:

Clinical variables
------------------

The following table provides the definitions of clinical variables. Each dataset will contain only a subset of these variables, dependent on what is available in the raw data.

.. list-table:: Clinical variables
   :widths: 20 80
   :header-rows: 1

   * - Abbreviation
     - Details
   * - age
     - Age (years)
   * - gender
     - Gender (female=1; male==2)
   * - weight
     - Weight (kg)
   * - height
     - Height (m)
   * - bmi
     - Body Mass Index (kg/m^2)
   * - dbp
     - Diastolic blood pressure (mmHg)
   * - sbp
     - Systolic blood pressure (mmHg)
   * - lvef
     - Left ventricular ejection fraction (LVEF) (%)
   * - prior_mi
     - Whether (1) or not (0) the subject has previously had a myocardial infarction (MI)
   * - nyha
     - New York Heart Association (NYHA) Functional Classification (1 = no limitation of physical activity; 2 = slight limitation; 3 = marked limitation; 4 = symptoms of heart failure at rest)
   * - diabetes
     - Whether (1) or not (0) the subject has diabetes.
   * - arb
     - Whether (1) or not (0) the subject is taking ARB (angiotensin II receptor blockers).
   * - ace_inhibitor
     - Whether (1) or not (0) the subject is taking ACE (angiotensin-converting enzyme) inhibitors.
   * - beta_blockers
     - Whether (1) or not (0) the subject is taking beta blockers.
   * - amiodarone
     - Whether (1) or not (0) the subject is taking amiodarone.
   * - no_pvcs_24hr
     - Number of premature ventricular contractions recorded in a 24-hour monitoring period.
   * - ischemic_dilated_cardiomyopathy
     - Whether (1) or not (0) the subject has ischemic dilated cardiomyopathy.
   * - hf
     - Whether (1) or not (0) the subject has heart failure
   * - sinusal rhythm
     - Whether (1) or not (0) the subject has sinusal rhythm.
   * - lvef_under_35
     - Whether (1) or not (0) the subject has a left ventricular ejection fraction (LVEF) of < 35%.
   * - nyha_class_iii
     - Whether (1) or not (0) the subject has a NYHA class of 3.
   * - arb_or_ace_inhibitors
     - Whether (1) or not (0) the subject is taking ARB (angiotensin II receptor blockers) or ACE (angiotensin-converting enzyme) inhibitors.
   * - temp
     - Body temperature (°C)
   * - heart_rate
     - Heart rate (beats per minute)
   * - resp_rate
     - Respiratory rate (breaths per minute)
   * - spo2
     - Blood oxygen saturation (%)  
   * - visited_ED_previously
     - Whether (1) or not (0) the subject had visited the Emergency Department (ED) previously.
   * - mi_visit_before
     - Whether (1) or not (0) the subject had a myocardial infarction (MI) diagnosis on a previous ED visit.
   * - cardiac_visit_before
     - Whether (1) or not (0) the subject had a cardiac diagnosis on a previous ED visit.
   * - cv_visit_before
     - Whether (1) or not (0) the subject had a cardiovascular diagnosis on a previous ED visit.
   * - any_visit_before
     - Whether (1) or not (0) the subject had any previous ED visit.
   * - cardiac_hosp_before
     - Whether (1) or not (0) the subject had been hospitalized for a cardiac reason before the index ED visit.
   * - cv_hosp_before
     - Whether (1) or not (0) the subject had been hospitalized for a cardiovascular reason before the index ED visit.
   * - hosp_before
     - Whether (1) or not (0) the subject had been hospitalized for any reason before the index ED visit.
   

.. _outcome-variables:

Outcome variables
-----------------

The following table provides the definitions of outcome variables. Each dataset will contain only a subset of these variables, dependent on what is available in the raw data.

.. list-table:: Outcome variables
   :widths: 20 80
   :header-rows: 1

   * - Abbreviation
     - Details
   * - survived
     - Whether or not the subject survived to the end of the follow-up time.
   * - other_death
     - Whether or not the subject died due to other causes during follow-up.
   * - sudden_cardiac_death
     - Whether or not the subject died due to sudden cardiac death during follow-up.
   * - pump_failure_death
     - Whether or not the subject died due to pump failure death during follow-up.
   * - \*_followup
     - Follow-up time (in years) for the outcome stated in place of \*.
   * - hosp_los
     - Length of hospital stay (in days).
   * - esi
     - Emergency Severity Index (ESI).
   * - admitted
     - Whether (1) or not (0) the subject was admitted to hospital.
   * - survived_hospital
     - Whether or not the subject survived to hospital discharge.
   * - visited_ED_subsequently
     - Whether (1) or not (0) the subject visited the Emergency Department (ED) visited_ED_subsequently
   * - icd10_diagnosis
     - ICD-10 diagnosis code assigned to the subject.
   * - cardiac_diagnosis
     - Whether (1) or not (0) the subject was given a cardiac diagnosis.
   * - cardiovascular_diagnosis
     - Whether (1) or not (0) the subject was given a cardiovascular diagnosis.
   * - mi_diagnosis
     - Whether (1) or not (0) the subject was given a myocardial infarction (MI) diagnosis.
   * - mi_visit_after
     - Whether (1) or not (0) the subject had a myocardial infarction (MI) diagnosis on a subsequent ED visit.
   * - time_to_mi_visit
     - Time to a subsequent ED visit at which the subject was given a myocardial infarction (MI) diagnosis.
   * - cardiac_visit_after
     - Whether (1) or not (0) the subject had a cardiac diagnosis on a subsequent ED visit.
   * - time_to_cardiac_visit
     - Time to a subsequent ED visit at which the subject was given a cardiac diagnosis.
   * - cv_visit_after
     - Whether (1) or not (0) the subject had a cardiovascular diagnosis on a subsequent ED visit.
   * - time_to_cv_visit
     - Time to a subsequent ED visit at which the subject was given a cardiovascular diagnosis.
   * - any_visit_after
     - Whether (1) or not (0) the subject had a subsequent ED visit.
   * - time_to_any_visit
     - Time to a subsequent ED visit.
   * - cardiac_hosp_after
     - Whether (1) or not (0) the subject was hospitalized for a cardiac reason after the index ED visit.
   * - time_to_cardiac_hosp
     - Time to hospitalization for a cardiac reason after the index ED visit.
   * - cv_hosp_after
     - Whether (1) or not (0) the subject was hospitalized for a cardiovascular reason after the index ED visit.
   * - time_to_cv_hosp
     - Time to hospitalization for a cardiovascular reason after the index ED visit.
   * - hosp_after
     - Whether (1) or not (0) the subject was hospitalized on a subsequent ED visit.
   * - time_to_hosp
     - Time to hospitalization following a subsequent ED visit.


.. _dataset-variables:

Dataset variables
------------------

The following table provides the definitions of dataset variables - those variables which describe what data are available in the dataset.

.. list-table:: Dataset variables
   :widths: 20 80
   :header-rows: 1

   * - Abbreviation
     - Details
   * - \*_available
     - Whether (1) or not (0) the signal abbreviated as \* is available in the recording
   * - holter_available
     - Whether (1) or not (0) there is a Holter recording file available in the dataset
   * - filetype
     - The filetype of the recording (e.g., 'WFDB', 'EDF', etc.)
   * - filepath
     - The filename of the recording (without an extension)
   * - \*_duration
     - The duration (in seconds) of the signal abbreviated as \* in the recording


.. _ecg-derived-variables:

ECG-derived variables
---------------------

The following table provides the definitions of ECG-derived variables - features derived from the ECG signals.

.. list-table:: ECG-derived variables
   :widths: 20 80
   :header-rows: 1

   * - Abbreviation
     - Details
   * - ecg\*_AC
     - The Phase Rectification Signal Averaging (PRSA) acceleration capacity (AC) derived from the ECG lead abbreviated as \*.
   * - ecg\*_DC
     - The Phase Rectification Signal Averaging (PRSA) deceleration capacity (DC) derived from the ECG lead abbreviated as \*.
   * - ecg\*_ECG_Rate_Mean
     - The mean heart rate (in beats per minute) derived from the ECG lead abbreviated as \*.
   * - ecg\*_HRV_\#
     - Heart Rate Variability (HRV) feature \# derived from the ECG lead abbreviated as \*. See the NeuroKit toolbox for details of HRV features.