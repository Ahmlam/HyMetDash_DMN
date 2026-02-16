#!/bin/bash

# Installer Google Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get update
apt-get install -y ./google-chrome-stable_current_amd64.deb

# Rendre Chrome disponible pour Kaleido
export CHROME_PATH="/usr/bin/google-chrome"

# S'assurer que le reste s'exécute bien
pip install -r requirements.txt
