#!/usr/bin/env bash
echo "Initializing virtualenv for Python3"
virtualenv --system-site-packages -p python3 ./
echo "Activating virtual env"
source bin/activate
echo "Installing requirements via pip"
pip install -r requirements.txt

echo "Installing spacy english model"
python -m spacy download en

echo "You must now pip install your prodigy wheel. Meet me at 'bootstrap_db.sh' when you're done."

