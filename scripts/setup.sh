#!/bin/bash

pip3 install -r requirements.txt

mkdir models
gdown 1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m 
mv bart_score.pth models/bart.pth # downloaded file should be in models folder