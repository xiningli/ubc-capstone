 Faulty-training-data Resiliency in Machine Learning Models using Ensemble Techniques 
=====================================================================================
By 
Xining Li (71361042) (xininglica@gmail.com) &  Debashis Kayal (12326609) (debkayal@student.ubc.ca)
-----------------------------------------------------------------------

Project Brief:
---------------

Ensemble learning is a machine learning approach that uses multiple algorithms to improve prediction quality. The original goal was to improve the inference accuracy of supervised learning by combining results from multiple decision makers. On the other hand, faults in the training data set are becoming commonplace. Are ensemble learning methods resilient to such problems?
This project involves an empirical analysis of ensemble learning approaches by subjecting the training data set to faults.

Steps To Execute The Code:
--------------------------- 
``` shell
1. Clone the git-repo
$ git clone https://github.com/xiningli/ubc-capstone.git 
The git clone will create a folder called "ubc-capstone"

2. Create a new folder under "ubc-capstone"
user@machine-os:/path/<gitrepofolder>/ubc-capstone$ mkdir all-data
user@machine-os:/path/<gitrepofolder>/ubc-capstone$ cd all-data

3. Copy and inflate the MNIST dataset from our Google Drive location using below commands (CIFAR10 is pulled by the code directly and WINE data set is within the git-repo)

user@machine-os:/path/<gitrepofolder>/ubc-capstone$ sudo wget -O all-data/minst.zip https://drive.google.com/u/0/uc\?id\=1b1-wmx02a5_5bWxXCNDOKwyhTJq0LnLM\&export\=download
user@machine-os:/path/<gitrepofolder>/ubc-capstone$ unzip all-data/minst.zip -d all-data/minst

4. Setup the environment (virtualenv)
Pre-requisites: python3.8, virtualenv, pip3
[for prerequisites:
 $ sudo apt-install python3.8
 $ virtualenv venv
 $ source venv/bin/activate]
 
(venv) user@machine-os:/path/<gitrepofolder>/ubc-capstone$ sh resolve-dependencies.sh

Note: The resolve dependencies shell will pull in  all required python libraries including some key libraries required to run our ml code such as: numpy,scikit-learn,matplotlib,pandas,xgboost,keras,tensorflow. Also it might take a while to complete.

5. Experiment #1: Runing the script for MNIST dataset
(venv) user@machine-os:/path/<gitrepofolder>/ubc-capstone$sh run-minst.sh

Runing the shell for WINE dataset
(venv) user@machine-os:/path/<gitrepofolder>/ubc-capstone$sh run-wine.sh

6. Experiment #2: Runing the script for Experiment #2. Pulls the CIFAR dataset (from  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
(venv) user@machine-os:/path/<gitrepofolder>/ubc-capstone$sh run-cifar.sh

```
The output screenshots are captured at https://github.com/xiningli/ubc-capstone/blob/master/OutputScreengrabs_RunningTheCode.docx



Additional Notes:
--------------------
1) Code File Name: Brief Description

wine/wine.py
minst/main.py
cifar10/main.py
cifar10-mislabeled/main.py

run-cifar.sh
run-minst.sh
run-wine.sh


