# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

all: mpi_kmeans

mpi_kmeans: mpi_kmeans.cu
	nvcc -I/usr/lib/openmpi/include -Iinc/ -L/usr/lib/openmpi/lib -lmpi mpi_kmeans.cu -o mpi_kmeans

clean:
	rm mpi_kmeans
