#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include "fonctions.h"

#include "matrice_vecteur.c"
#include "csvparser.h"
#include "csvparser.c"
#include "csvwriter.h"
#include "csvwriter.c"

int main(int argc, char *argv[]){
    int min = 0;
    int max = 1;
    int dim = atoi(argv[1]);
    int compteur;
    int keep_on = 1;

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double *matrix;
    double *vector;
    double *last_vector;
    double error_max;
    double error;
    double *ones;
    double damping;

    if(rank==0){
        compteur=0;
        matrix = malloc(dim*dim*sizeof(double)); vector = malloc(dim*sizeof(double)); last_vector = malloc(dim*sizeof(double)); ones = malloc(dim*dim*sizeof(double));
        error_max = 1.0/(100000.0);
        damping = 0.85;
        fill_with_ones(ones,dim); fill_randomly(vector,dim,1,min,max); fill_randomly(last_vector,dim,1,0,0);  fill_randomly(matrix,dim,dim,min,max);
        normalize_matrix(matrix,dim); normalize_vector(vector,dim);
        scalar_multiply(matrix,dim,damping,matrix); scalar_multiply(ones,dim,(1-damping)/(1.0*dim),ones);
        add(matrix,ones,dim,matrix);
        printf("la matrice est : \n");fflush(stdout);
        print_matrix(matrix,dim,dim);
        sum_column(matrix,dim);
    }

    while(keep_on==1){
        if (rank==0){
            compteur+=1;
            fill(last_vector,vector,dim);
        }
        compute_product_matrix_vector_parallel(matrix, vector, dim, dim, MPI_COMM_WORLD, status, rank, size, argc, argv, vector);
        if (rank==0){
            error = eucl_dist(vector,last_vector,dim);
            if (error<error_max){
                keep_on = 0;
            }
        }
        MPI_Bcast(&keep_on,1,MPI_INT,0,MPI_COMM_WORLD);
   }

    if(rank==0){
        printf("Il y a eu %d iterations \n",compteur);
        printf("error=%f, error max = %f\n ", eucl_dist(vector,last_vector,dim),error_max);
        printf("Norme 1 du resultat = %f \n",norm1(vector,dim));
    }

    MPI_Finalize();
    free(matrix); free(vector); free(ones);free(last_vector);
    return 0;
}
