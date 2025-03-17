Requires Python 3.10 and pip to be running on python 3.10

If you have a python version less than 3.10 then run the below command 
brew install python@3.10 

If your python command defaults to a python version less than 3.10, then when using the python command to run the python files use python3.10 instead

If Pip is not running on python3.10 or higher than run the below commands (you can check with pip --version)
python3.10 -m ensurepip
python3.10 -m pip install --upgrade pip

First install requirements 
pip install -r requirements.txt

To run each of the 3 files
python3.10 k-means_uri.py
python3.10 k-means_kmi.py
python3.10 hierarchical_agglomerative.py

This code heavily leverages the scikit-learn module to create the hierarchical agglomerative clustering method, matplotlib to generate plots for each of the 4 models, and scipy to create the dendrograms for the hierarchical agglomerative models. 