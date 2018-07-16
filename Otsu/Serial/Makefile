all: serial

serial: serial_otsu.cu
	nvcc -Iinc/ -o serial_otsu serial_otsu.cu

clean:
	rm serial_otsu
