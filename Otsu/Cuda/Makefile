all: cuda

cuda: cuda_otsu.cu
	nvcc -Iinc/ -o cuda_otsu cuda_otsu.cu

clean:
	rm cuda_otsu
