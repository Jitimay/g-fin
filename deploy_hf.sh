#!/bin/bash
# Replace YOUR_USERNAME and SPACE_NAME with your actual values
git clone https://huggingface.co/spaces/Jitimay/gfin-financial-immunity
cd SPACE_NAME

# Copy and rename files
cp ../app_hf.py app.py
cp ../requirements_hf.txt requirements.txt
cp ../README_HF.md README.md

# Copy data and model directories
cp -r ../data .
cp -r ../model .

# Deploy
git add .
git commit -m "Deploy G-FIN Financial Immunity Network"
git push
