#! /usr/bin/python

import os
if not os.path.exists("./all-data/minst.zip"):
    os.system('wget -O all-data/minst.zip https://drive.google.com/u/0/uc\?id\=1b1-wmx02a5_5bWxXCNDOKwyhTJq0LnLM\&export\=download')
if not os.path.exists("./all-data/minst"):
    os.system("unzip all-data/minst.zip -d all-data/minst")

os.system("sh resolve-dependencies.sh")
os.system("sh run-minst.sh")
os.system("sh generate-pdf.sh")
