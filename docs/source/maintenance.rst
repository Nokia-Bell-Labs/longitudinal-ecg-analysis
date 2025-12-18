Maintenance
===========

This page provides details of how to maintain the repository.

Generate Documentation
----------------------

The documentation is a mixture of manually written pages, and auto-generated API documentation. 

* Update documentation manually by editing _.rst_ files in ``longitudinal_ecg_analysis/docs/source/``

  * Documentation is written in ``reStructuredText`` syntax, as documented `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ .
  * If adding a new file, be sure to include it in the table of contents in ``index.rst``

* Use a virtual environment with the required dependencies for building docs, e.g. created using::
   
   cd longitudinal-ecg-analysis/
   python3 -m venv .venv-docs
   source .venv-docs/bin/activate
   pip install --uppgrade pip
   pip install -r docs/requirements.txt

* Test build the docs locally using::
   
   cd docs
   sphinx-apidoc -f -o source/api ../src/longitudinal_ecg_analysis
   sphinx-build -b html source build/html

* Raise a PR, and then the docs will be built using a GitHub action.

   
Generate requirements.txt
-------------------------

After modifying the code, run the following command to update requirements.txt::

    pip freeze > requirements.txt