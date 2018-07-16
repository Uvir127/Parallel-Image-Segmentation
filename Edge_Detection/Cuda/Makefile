all: cuda_edge_detection

cuda_edge_detection: cuda_edge_detection.cu
	nvcc -Iinc/ -o cuda_edge_detection cuda_edge_detection.cu

clean:
	rm cuda_edge_detection
