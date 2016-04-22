#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include "fonctions.h"
#include <time.h>

double random(double min, double max){
    double r = (double)1.0*rand()/(1.0*RAND_MAX);
    return min + r*(max-min);
}

void fill_randomly(double *matrix, int line_number, int column_number, int min_value, int max_value){
    int i = 0;
    int j = 0;
    for(i=0;i<line_number;i++){
        for(j=0;j<column_number;j++){
            matrix[i*column_number+j] = random(min_value,max_value);
        }
    }
    //print_matrix(matrix,line_number,column_number);
}

void print_matrix(double *matrix, int number_of_rows, int number_of_columns){
    int i;
    int j;
    for(i=0;i<number_of_rows;i++){
        for(j=0;j<number_of_columns;j++){
            printf("%f ", matrix[i*number_of_columns + j]); fflush(stdout);
        }
        printf("\n");fflush(stdout);
    }
    printf("\n\n");fflush(stdout);
}

void compute_product_matrix_vector(double *matrix, int row_number, int column_number, double *vector, double *result){ /* could be extended to compute matrix matrix product
                                                                                    Doesn't try to manage what happens where the sizes are invalid (too big or too small)...*/
    int i = 0;
    int j = 0;
    double *clone;
    clone=malloc(sizeof(double)*row_number);
    for(i=0;i<row_number;i++){
        clone[i] = vector[i];
    }
    for(i=0;i<row_number;i++){
        result[i] = 0;
        for(j=0;j<column_number;j++){
            result[i] += matrix[i*column_number + j]*clone[j];
        }
    }
    free(clone);
    printf("\n\n");
}

void fill(double *vector_to_be_filled, double *filling_vector, int dim){
    int i;
    for (i=0;i<dim;i++){
        vector_to_be_filled[i] = filling_vector[i];
    }
}

void fill_locally(double *result, double *local_result, int my_row_number, int shift){
    int i;
    for(i=0;i<my_row_number;i++){
        result[i + shift] = local_result[i];
    }
}

void fill_with_zeros(double *vector, int dim){
    int i;
    for(i=0;i<dim;i++){
        vector[i] = 0;
    }
}

void fill_with_ones(double *A, int dim){
    int i,j;
    for (i=0;i<dim;i++){
        for(j=0;j<dim;j++){
            A[i*dim+j] = 1;
        }
    }
}

void normalize_matrix(double* matrix, int dim){ // from a L1 norm perspective
    double s;
    int i,j;
    for(j=0;j<dim;j++){
        s= 0;
        for(i=0;i<dim;i++){
            s+=matrix[i*dim+j];
        }
        for(i=0;i<dim;i++){
            matrix[i*dim+j] = matrix[i*dim+j]/s;
        }
    }
}

void normalize_vector( double* vector, int dim){ // from a L1 norm perspective
    double s=0;
    int i;
    for(i=0;i<dim;i++){
        s += vector[i];
    }
    for(i=0;i<dim;i++){
        vector[i] = vector[i]/s;
    }
}

void scalar_multiply(double *A, int dim, double coeff, double *result){
    int i,j;
    for (i=0;i<dim;i++){
        for(j=0;j<dim;j++){
            result[i*dim+j] = coeff*A[i*dim+j];
        }
    }
}

void add(double *A, double *B, int dim, double *result){
    int i,j;
    for (i=0;i<dim;i++){
        for(j=0;j<dim;j++){
            result[i*dim+j] = A[i*dim+j]+B[i*dim+j];
        }
    }
}

double norm1(double *vector, int dim){ // L1 norm
    double s = 0;
    int i;
    for(i=0;i<dim;i++){
        if(vector[i]>0){s += vector[i];} else{s -= vector[i];}
    }
    return s;
}

double eucl_dist(double *a, double *b, int dim){
    double s = 0;
    int i;
    for(i=0;i<dim;i++){
        s += (a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(s);
}

void sum_column(double *A, int dim){
    double *vector;
    vector = malloc(dim*sizeof(double));
    int i,j;
    for(j=0;j<dim;j++){
        vector[j]=0;
        for(i=0;i<dim;i++){
            vector[j]+=A[i*dim+j];
        }
    }
    print_matrix(vector,dim,1);
}

void compute(double *local_matrix, double *local_vector, int local_vector_dim, int row_number, int column_number, int shift, double *local_result){
    int i,j;
    for(i=0;i<row_number;i++){
        for(j=0;j<local_vector_dim;j++){
            local_result[i] += local_matrix[column_number*i + shift + j]*local_vector[j];
        }
    }
}

void compute_product_matrix_vector_parallel(double *matrix, double *vector, int row_number, int column_number, MPI_Comm communicator, MPI_Status status, int rank, int size, int argc, char *argv[], double *result){ // the result will be stored in result
    int is_slave;
    if (rank==0){
        is_slave=0;
    }
    else{
        is_slave=1;
    }
    MPI_Comm slave_communicator; MPI_Comm_split(communicator, is_slave, rank, &slave_communicator);
    int slave_rank, slave_size;
    MPI_Comm_rank(slave_communicator, &slave_rank);
    MPI_Comm_size(slave_communicator, &slave_size);

    int local_row_number = row_number/(size-1); int last_row_number = local_row_number + row_number%(size-1);

    double *local_matrix,*local_vector,*tmp_vector,*local_result;
    int tmp_row_number;

    int source,tag,dest,root,shift; tag=1;

    // 1) Allocates memory to local variables
    if(rank!=0){
        if(rank<size-1){
            local_matrix = malloc(sizeof(double)*local_row_number*column_number);
            local_vector = malloc(sizeof(double)*local_row_number);
            local_result = malloc(sizeof(double)*local_row_number);
            fill_with_zeros(local_result,local_row_number);
        }
        else{
            local_matrix = malloc(sizeof(double)*last_row_number*column_number);
            local_vector = malloc(sizeof(double)*last_row_number);
            local_result = malloc(sizeof(double)*last_row_number);
            fill_with_zeros(local_result,last_row_number);
        }
    }

    // 2) Sends local variable to processors 1...size-1
    if(rank==0){
        for(dest=1;dest<size-1;dest++){
            local_matrix = matrix +  column_number*(dest-1)*local_row_number;
            local_vector = vector + (dest-1)*local_row_number;
            MPI_Send(local_vector, local_row_number, MPI_DOUBLE, dest, tag, communicator);
            MPI_Send(local_matrix, local_row_number*column_number, MPI_DOUBLE, dest, tag, communicator);
        }
        dest = size-1;
        local_matrix = matrix +  column_number*(dest-1)*local_row_number;
        local_vector = vector + (dest-1)*local_row_number;
        MPI_Send(local_vector, last_row_number, MPI_DOUBLE, dest, tag, communicator);
        MPI_Send(local_matrix, last_row_number*column_number, MPI_DOUBLE, dest, tag, communicator);
    }

    // 3) Receives the local matrix and vector from master
    else{
        source = 0;
        if(rank<size-1){
            MPI_Recv(local_vector, local_row_number, MPI_DOUBLE, source, tag, communicator,&status);
            MPI_Recv(local_matrix, column_number*local_row_number, MPI_DOUBLE, source, tag, communicator,&status);
        }
        else{
            MPI_Recv(local_vector, last_row_number, MPI_DOUBLE, source, tag, communicator,&status);
            MPI_Recv(local_matrix, column_number*last_row_number, MPI_DOUBLE, source, tag, communicator,&status);
        }

    }

    // 4) Computes the product
    if (rank!= 0){
        tmp_vector = malloc(sizeof(double)*last_row_number);
        if(rank<size-1){
            tmp_row_number = local_row_number;
        }
        else{
            tmp_row_number = last_row_number;
        }
        fill(tmp_vector,local_vector,tmp_row_number);
        for(root=0;root<slave_size;root++){
            if(root==slave_rank){ // in that case, use your own
                if(rank<size-1){
                    tmp_row_number = local_row_number;
                }
                else{
                    tmp_row_number = last_row_number;
                }
                fill(tmp_vector, local_vector, tmp_row_number);
            }
            MPI_Bcast(tmp_vector,last_row_number,MPI_DOUBLE,root,slave_communicator);
            MPI_Bcast(&tmp_row_number,1,MPI_INT,root,slave_communicator);
            if (rank<size-1){
                compute(local_matrix, tmp_vector, tmp_row_number, local_row_number, column_number, root*local_row_number, local_result);
            }
            else{
                compute(local_matrix, tmp_vector, tmp_row_number, last_row_number, column_number, root*local_row_number, local_result);
            }
        }
        dest=0;
        if (rank<size-1){
            MPI_Send(local_result, local_row_number, MPI_DOUBLE, dest, tag, communicator);
        }
        else{
            MPI_Send(local_result, last_row_number, MPI_DOUBLE, dest, tag, communicator);
        }
        free(local_result); free(local_vector); free(tmp_vector);//free(local_matrix);
    }


    // 5) Receives the results from the slaves and prints them
    else{
        local_result = malloc(sizeof(double)*local_row_number);
        for(source=1;source<size-1;source++){
            shift = (source-1)*local_row_number;
            MPI_Recv(local_result, local_row_number, MPI_DOUBLE, source, tag, communicator,&status);
            fill_locally(result,local_result,local_row_number,shift);
        }
        source = size-1;
        shift = (source-1)*local_row_number;
        free(local_result);
        local_result = malloc(sizeof(double)*last_row_number);
        MPI_Recv(local_result, last_row_number, MPI_DOUBLE, source, tag, communicator,&status);
        fill_locally(result,local_result,last_row_number,shift);
        printf("The result is : \n"); fflush(stdout); print_matrix(result,row_number,1);
        free(local_result);
    }

}



int main(int argc, char *argv[]){
    srand(time(NULL));
    int min_value_in_matrix = 0;
    int max_value_in_matrix = 10;
    int row_number = atoi(argv[1]);
    int column_number = atoi(argv[2]);

    // generates a matrix
    double *matrix ;
    double *vector ;
    double *result;
    double *ones;
    double d;

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank==0){
        d=0.85;
        ones = malloc(sizeof(double)*row_number*column_number); fill_with_ones(ones, row_number);
        result=malloc(sizeof(double)*row_number);
        matrix = malloc(row_number*column_number*sizeof(double));
        fill_randomly(matrix, row_number, column_number, min_value_in_matrix, max_value_in_matrix);
        vector = malloc(row_number*sizeof(double));
        fill_randomly(vector, row_number, 1, min_value_in_matrix, max_value_in_matrix);
        fill_randomly(result, row_number, 1, 0, 0);
        normalize_matrix(matrix,row_number); normalize_vector(vector,row_number);
        scalar_multiply(matrix,row_number,d,matrix); scalar_multiply(ones,row_number,(1-d)/(1.0*row_number),ones);
        add(matrix,ones,row_number,matrix);
	}

    compute_product_matrix_vector_parallel(matrix, vector, row_number, column_number, MPI_COMM_WORLD, status, rank, size, argc, argv, result);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    free(result); free(matrix); free(vector);
    return 0;
}



