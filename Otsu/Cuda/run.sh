#!/bin/bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
make
./cuda_otsu
