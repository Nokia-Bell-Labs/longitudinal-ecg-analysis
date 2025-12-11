Examples
========

.. _running-demo-analysis:

Running demo analysis
---------------------

Use ``longitudinal_ecg_analysis.run_demo`` to run a short demo analysis.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.run_demo <dataset_root_raw_folder> <dataset_root_proc_folder> <dataset_name>

where:
* `dataset_root_raw_folder` is the root folder for the raw data.
* `dataset_root_proc_folder` is the root folder for the processed data.
* `dataset_name` is the name of the dataset to be processed (e.g. music).

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.run_demo_analysis /Users/petercharlton/Documents/data/music/raw /Users/petercharlton/Documents/data/music/proc music

This command will perform all of the steps outlined on the :ref:`Overview page <steps>`. The following examples cover each of these steps individually.


.. _generating-dataset-settings:

Generating Dataset Settings
---------------------------

Use ``longitudinal_ecg_analysis.gen_dataset_settings`` to generate a JSON settings file for a particular dataset.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.gen_dataset_settings <dataset_root_raw_folder> <dataset_root_proc_folder> <dataset_name>

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.gen_dataset_settings /Users/petercharlton/Documents/data/music/raw /Users/petercharlton/Documents/data/music/proc music


.. _curating-entire-dataset:

Curating Entire Dataset
-----------------------

Use the ``longitudinal_ecg_analysis.curate_dataset`` module to curate an entire dataset.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.curate_entire_dataset <dataset_root_proc_folder> [--redo_everything]

where:
* `--redo_everything` is an optional flag to re-process all data, even if already processed.

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.curate_entire_dataset /Users/petercharlton/Documents/data/music/proc


.. _generating-analysis-settings:

Generating Analysis Settings
----------------------------

Use ``longitudinal_ecg_analysis.gen_settings`` to generate a JSON settings file for your analysis.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.gen_dataset_settings <dataset_root_proc_folder> <analysis_name> <do_demo>

where:
* `analysis_name` is the name of the analysis to be performed (e.g. analysis1).
* `do_demo` is a flag indicating whether to run a demo analysis (e.g. True or False).

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.gen_settings /Users/petercharlton/Documents/data/music/raw /Users/petercharlton/Documents/data/music/proc music analysis1


.. _curating-analysis-dataset:

Curating Analysis Dataset
-------------------------

Use ``longitudinal_ecg_analysis.curate_analysis_dataset`` to curate a subset of a dataset for analysis.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.curate_analysis_dataset <dataset_root_proc_folder> <analysis_name>

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.curate_analysis_dataset /Users/petercharlton/Documents/data/music/proc analysis1


.. _deriving-signal-features:

Deriving signal features
------------------------

Use ``longitudinal_ecg_analysis.derive_signal_features`` to derive signal features from ECG signals.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.derive_signal_features <dataset_root_proc_folder> <analysis_name>

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.derive_signal_features /Users/petercharlton/Documents/data/music/proc analysis1


.. _compiling-for-stats:

Compiling for statistical analysis
----------------------------------

Use ``longitudinal_ecg_analysis.compile_for_stats`` to compile derived features into aggregate metrics for statistical analysis.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.compile_for_stats <dataset_root_proc_folder> <analysis_name>

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.compile_for_stats /Users/petercharlton/Documents/data/music/proc analysis1


.. _statistical_analysis:

Performing statistical analysis
-------------------------------

Use ``longitudinal_ecg_analysis.stats_analysis`` to investigate associations between the derived features and health outcomes.

Usage
^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.stats_analysis <dataset_root_proc_folder> <analysis_name>

Example
^^^^^^^

.. code-block:: bash

   python -m longitudinal_ecg_analysis.stats_analysis /Users/petercharlton/Documents/data/music/proc analysis1