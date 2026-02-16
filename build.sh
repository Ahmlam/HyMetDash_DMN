#!/usr/bin/env bash
set -o errexit

# Installer Chrome requis par Kaleido
apt-get update && apt-get install -y wget gnupg unzip
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get install -y ./google-chrome-stable_current_amd64.deb || apt-get -f install -y

# Variables d'environnement pour Kaleido
export DISPLAY=:99.0
