#!/bin/bash

# This script deploys the G-FIN Premium Terminal to Hugging Face Spaces.

# --- Configuration ---
# Your Hugging Face username
HF_USERNAME="Jitimay"
# The name of your Hugging Face Space
SPACE_NAME="gfin-financial-immunity"
# The local directory where you will clone the space repository
CLONE_DIR="gfin-financial-immunity-space"
# The commit message
COMMIT_MESSAGE="Deploy G-FIN Premium Terminal"

# --- End of Configuration ---

# --- Safety Checks ---
# Check if git is installed
if ! command -v git &> /dev/null
then
    echo "Error: git is not installed. Please install git to use this script."
    exit 1
fi

# --- Deployment Steps ---

echo "--- Starting Deployment to Hugging Face Spaces ---"

# 1. Clone the Hugging Face Space repository
# Remove the directory if it already exists to start fresh
if [ -d "$CLONE_DIR" ]; then
    echo "Removing existing clone directory: $CLONE_DIR"
    rm -rf "$CLONE_DIR"
fi

echo "Cloning repository from Hugging Face..."
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" "$CLONE_DIR"

# Check if cloning was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone the repository. Please check the username, space name, and your permissions."
    exit 1
fi

# 2. Navigate into the cloned repository
cd "$CLONE_DIR"

# 3. Initialize Git LFS and track large files
echo "Initializing Git LFS and tracking large model files..."
git lfs install
git lfs track "model/*.pkl"

# 4. Create the README.md file with the correct configuration
echo "Creating README.md for Streamlit app configuration..."
cat > README.md << EOL
---
title: G-FIN Terminal Premium
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
---

# G-FIN - Global Financial Immunity Network (Premium)

This is the premium version of the G-FIN Terminal, an AI-powered financial crisis prediction system.
EOL

# 4. Copy the application files
echo "Copying application files..."

# Copy the main app file and rename it
cp ../app_premium.py app.py
if [ $? -ne 0 ]; then echo "Error: app_premium.py not found."; exit 1; fi

# Copy the requirements file
cp ../requirements.txt requirements.txt
if [ $? -ne 0 ]; then echo "Error: requirements.txt not found."; exit 1; fi

# Copy the data and model directories
cp -r ../data .
if [ $? -ne 0 ]; then echo "Error: data directory not found."; exit 1; fi

cp -r ../model .
if [ $? -ne 0 ]; then echo "Error: model directory not found."; exit 1; fi

# 5. Commit and push the files to Hugging Face
echo "Adding files to git, committing, and pushing..."

git add .
git commit -m "$COMMIT_MESSAGE"
git push

# Check if push was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to push to Hugging Face. Please check your credentials and permissions."
    exit 1
fi

echo "--- Deployment Successful! ---"
echo "Your application is now being built on Hugging Face Spaces."
echo "You can view the status at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

# Go back to the original directory
cd ..
