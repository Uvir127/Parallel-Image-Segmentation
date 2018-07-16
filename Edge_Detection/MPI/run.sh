#!/bin/bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
make
mpiexec -n 8 mpi_edge_detection
