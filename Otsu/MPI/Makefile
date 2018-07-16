# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

all: mpi

mpi: mpi_otsu.cu
	nvcc -I/usr/lib/openmpi/include -Iinc/ -L/usr/lib/openmpi/lib -lmpi mpi_otsu.cu -o mpi_otsu

clean:
	rm mpi_otsu
