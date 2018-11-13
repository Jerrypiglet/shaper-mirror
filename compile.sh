#!/usr/bin/env bash
conda activate shaper
if [ ! -d "build" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir build
fi
cd build
cmake .. && make
