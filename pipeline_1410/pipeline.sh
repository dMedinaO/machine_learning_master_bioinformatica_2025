#!/usr/bin/bash

python get_properties.py ../raw_data/demo_amp.csv sequence ../results/1410/described_dataset.csv

python prepare_dataset.py ../configs/config_model_1410.json

python training_under_exploring.py ../results/1410/ label ../configs/config_hyperparams_models.json

