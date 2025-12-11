Maintenance
===========

This page provides details of how to maintain the repository.

Generate Documentation
----------------------

The documentation is a mixture of manually written pages, and auto-generated API documentation. 

* Update documentation manually by editing _.rst_ files in ``longitudinal_ecg_analysis/docs/source/``

  * Documentation is written in ``reStructuredText`` syntax, as documented `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ .
  * If adding a new file, be sure to include it in the table of contents in ``index.rst``

* Use a virtual environment, e.g. created using::
   
   cd longitudinal-ecg-analysis/
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

* Update the auto-generated API documentation using::
   
   cd longitudinal-ecg-analysis/
   sphinx-apidoc -f -o docs/source ./src/longitudinal_ecg_analysis
   cd docs
   make html
   cd ..

Generate requirements.txt
-------------------------

After modifying the code, run the following command to update requirements.txt::

    pip freeze > requirements.txt