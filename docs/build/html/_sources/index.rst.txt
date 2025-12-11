.. longitudinal-ECG-analysis documentation master file

Longitudinal ECG Analysis
=========================

A package to investigate the associations between ECG features and health outcomes.

Features
--------

- Derives features from ECG signals
- Performs statistical analysis of associations between ECG features and health outcomes
- Compatible with two publicly available datasets

Installation
------------

.. code-block:: bash

    pip install git+https://github.com/Nokia-Bell-Labs/longitudinal-ECG-analysis


.. _quick-example:

Quick Example
-------------

* Download the :ref:`MUSIC Dataset <music-dataset>`:

  * To run the demo analysis, download the files in the root folder (such as `subject-info.csv`), and download `Holter_ECG` files for the first 25 subjects.
  * Store these in a folder, and note down the path of this folder, which will be used as the `<raw_data_folder>`.

* Create a folder in which to store the results, and note down the path of this folder, which will be used as the `<processing_folder>`.

* Clone the repository:

.. code-block:: bash

    git clone https://github.com/Nokia-Bell-Labs/longitudinal-ECG-analysis.git

* Install the required packages, preferably in a virtual environment, using:

.. code-block:: bash

    cd longitudinal-ECG-analysis
    pip install -r requirements.txt

* Run the demo analysis using:

.. code-block:: python

    cd longitudinal_ecg_analysis/src
    python -m longitudinal_ecg_analysis.run_demo <raw_data_folder> <processing_folder> music

Further details of this example are provided in the :ref:`Running demo analysis <running-demo-analysis>` example.

------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   overview
   datasets
   examples
   variables
   maintenance
   API Documentation <modules>
   