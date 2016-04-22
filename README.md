# parallel_product_matrix_vector
A C implementation of an algorithm for parallel product matrix vector computation. Uses MPICH2 library
A lot of mathematical problems can be reduced to the solving of a big linear system. It's the case for PDE solving for instance. Often, you have to perform the basic operation of multiplying matrices with vectors. When the size of the matrices to be multiplied gets very large, it becomes interesting to compute the product in parallel. 

Each processor contains a raw band of the matrix and the vector. They exchange those data when they compute the product. It allows not to store too many data on one given processor, since it has to 


