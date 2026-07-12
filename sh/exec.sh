#!/bin/bash
cd "${HOME}/BoatRacePredictions"
nohup env OPTIMIZE_RERANK_WORKERS=8 bash sh/train_full.sh > script.log 2>&1 &
