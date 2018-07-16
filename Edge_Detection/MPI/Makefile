# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

all: mpi_edge_detection

mpi_edge_detection: mpi_edge_detection.cu
	nvcc -I/usr/lib/openmpi/include -Iinc/ -L/usr/lib/openmpi/lib -lmpi mpi_edge_detection.cu -o mpi_edge_detection

clean:
	rm mpi_edge_detection
