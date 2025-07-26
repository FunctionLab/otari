#!/bin/sh

# Ensure we're in the otari directory
cd ~/otari || { echo "Directory otari does not exist. Please clone the directory and try again."; exit 1; }

# Otari resources
wget https://zenodo.org/records/16433545/files/otari_resources.tar.gz

# Extract into ./resources within ./otari
tar -xzvf otari_resources.tar.gz

# Trained Otari model (extract into ./resources)
wget -P resources/ https://zenodo.org/records/16432270/files/otari.pth.gz
