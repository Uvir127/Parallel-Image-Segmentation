#!/bin/bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
make
./serial_edge_detection
