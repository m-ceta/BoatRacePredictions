#!/bin/bash
cd ${HOME}/BoatRacePredictions
nohup bash sh/train_full.sh > script.log 2>&1 &
