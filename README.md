# parallel_product_matrix_vector
A C implementation of an algorithm for parallel product matrix vector computation. Uses MPICH2 library
A lot of mathematical problems can be reduced to the solving of a big linear system. It's the case for PDE solving for instance. Often, you have to perform the basic operation of multiplying matrices with vectors. When the size of the matrices to be multiplied gets very large, it becomes interesting to compute the product in parallel. 

Each processor contains a raw band of the matrix and the vector. They exchange those data when they compute the product. It allows not to store too many data on one given processor, since it has to store them in its cache. If there is too much data, it can't and it makes the computation longer.

A function allows you to compute the product using MPICH2
There is a main method with an example

There is also a page_rank file that is an application of the last algorithm. See https://en.wikipedia.org/wiki/PageRank (it uses the power method)

To run the code :
mpiexec –localonly [number_of_processors] matrice_vecteur.exe [row_number_of_matrix] [column_number_of_matrix] 
