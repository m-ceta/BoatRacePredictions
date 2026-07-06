#!/bin/bash
cd ${HOME}/BoatRacePredictions
nohup bash sh/gcloud_debian_full_pipeline.sh > script.log 2>&1 &
