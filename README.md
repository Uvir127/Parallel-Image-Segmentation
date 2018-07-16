# Parallel-Image-Segmentation
This project contains 3 different image segmentation algorithms parallelised using CUDA and MPI. The 3 chosen image segmentation algorithms are Edge Detection, Otsu Thresholding and K-Means Clustering

Each implementation will work for any image. (Has to be colour image for K-Means)
All images used must be of format pgm for grayscale images and ppm for rgb images.
A makefile is included in each implemenation of each algorithm.
A run.sh file is also included.
Any number of nodes can be used in MPI.

----------------------Otsu----------------------
Any grayscale image can be used.
To change the image, the image must be in the data folder.
The image name must be changed in the otsu file.

-----------------Edge-Detecion------------------
Any grayscale image can be used.
To change the image, the image must be in the data folder.
The image name must be changed in the edge_detection file.
When changing the filter, change the filter size as well.

-----------------K-Means------------------------
Any RGB image can be used.
To change the image, the image must be in the data folder.
The image name must be changed in the k_means file.
