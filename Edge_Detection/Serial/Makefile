all: serial_edge_detection

serial_edge_detection: serial_edge_detection.cu
	nvcc -Iinc/ -o serial_edge_detection serial_edge_detection.cu

clean:
	rm serial_edge_detection
