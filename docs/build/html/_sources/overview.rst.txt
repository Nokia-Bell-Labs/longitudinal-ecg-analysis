Overview
========


The **Longitudinal ECG Analysis** package is designed to investigate the associations between ECG features and health outcomes.

It provides a processing pipeline to extract features from longitudinal ECG signals and investigate the associations between these features and health outcomes.

.. image:: _static/package_flowchart.png
   :alt: The **Longitudinal ECG Analysis** package
   :align: center
   :width: 600px


Image sources: `Mika Baumeister on Unsplash <https://unsplash.com/photos/white-printing-paper-with-numbers-Wpnoqo2plFA>`_; `Joshua Chehov on Unsplash <https://unsplash.com/photos/a-green-heart-beat-on-a-black-background-oCSol-lBtVA>`_; `Elen Sher on Unsplash <https://unsplash.com/photos/a-woman-sitting-at-a-desk-using-a-computer-0dF7UzD2Yd8>`_; `Ian Taylor on Unsplash <https://unsplash.com/photos/yellow-and-white-van-on-road-during-daytime-4hWvAJP8ofM>`_.

.. _steps:

Steps
-----

The **Longitudinal ECG Analysis** package performs the following steps:

#. :ref:`Generate dataset settings <generating-dataset-settings>`: *create a text file specifying the settings for a particular dataset*
#. :ref:`Curate entire dataset <curating-entire-dataset>`: *curate an entire dataset to get it ready for analysis*
#. :ref:`Generate analysis settings <generating-analysis-settings>`: *create a text file specifying the analysis settings*
#. :ref:`Curate analysis dataset <curating-analysis-dataset>`: *extract and curate a subset of a dataset for analysis*
#. :ref:`Derive signal features <deriving-signal-features>`: *derive features from ECG signals*
#. :ref:`Compile for stats <compiling-for-stats>`: *compile features ready for statistical analysis*
#. :ref:`Perform statistical analysis <statistical_analysis>`: *investigate the associations between features and health outcomes*
