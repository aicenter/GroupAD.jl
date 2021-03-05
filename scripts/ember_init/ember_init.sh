#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# create a separate py envronment
python3 -m venv ember-env
source ember-env/bin/activate
pip install git+https://github.com/elastic/ember.git
python3 $SCRIPTPATH/ember_init.py
#cleanup
deactivate
rm -r ember-env