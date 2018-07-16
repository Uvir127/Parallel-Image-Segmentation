# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

all: serial_kmeans

serial_kmeans: serial_kmeans.cu
	nvcc -I/usr/lib/openmpi/include -Iinc/ -L/usr/lib/openmpi/lib -lmpi serial_kmeans.cu -o serial_kmeans

clean:
	rm serial_kmeans
