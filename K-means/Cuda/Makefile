# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

all: cuda_kmeans

cuda_kmeans: cuda_kmeans.cu
	nvcc -I/usr/lib/openmpi/include -Iinc/ -L/usr/lib/openmpi/lib -lmpi cuda_kmeans.cu -o cuda_kmeans

clean:
	rm cuda_kmeans
