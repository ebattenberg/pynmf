
#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"
#include<time.h>
#include<sys/time.h>
#include<string.h>

#include<omp.h>
#include<mkl.h>


#define MAX_ITER 100
#define TRIALS 10
#define MKL_THREADS 2
#define MAT_DIM 100


double get_time();

int mkl_threads;
int matrix_dim[3];

int main(int argc, char* argv[]){


    //factor X into W*H
    matrix A,B,C;
    


    int max_iter;
    if(argc > 1){
	if(!strcmp(argv[1],"-h")){
	    printf("usage: bench matrix_dim1[100] matrix_dim2[100] matrix_dim3[100] iterations[100] trials[10] mkl_threads[#procs]\n");
	    exit(0);	
	}
    }
    if (argc > 3){
	    matrix_dim[0] =  atoi(argv[1]);
	    matrix_dim[1] =  atoi(argv[2]);
	    matrix_dim[2] =  atoi(argv[3]);
    }
    else {
	matrix_dim[0] = MAT_DIM;
	matrix_dim[1] = MAT_DIM;
	matrix_dim[2] = MAT_DIM;
    }

    int num_trials;
    if(argc>5)
	num_trials = atoi(argv[5]);
    else 
	num_trials = TRIALS;

    int verbose = 0;
    if (num_trials<10)
	verbose = 1;


    mkl_threads = mkl_get_max_threads();
    if (argc>6){
	if (atoi(argv[6]) < mkl_threads)
	    mkl_threads = atoi(argv[6]);
    }

    if (argc>4)
	max_iter = atoi(argv[4]);
    else
	max_iter = MAX_ITER;


    printf("mkl_threads: \t\t%i\n",mkl_threads);
    printf("matrix_dims: \t\t%i,%i,%i\n",matrix_dim[0],matrix_dim[1],matrix_dim[2]);


    create_matrix(&A, matrix_dim[0], matrix_dim[1], 1);
    create_matrix(&B, matrix_dim[1], matrix_dim[2], 1);
    create_matrix(&C, matrix_dim[0], matrix_dim[2], 1);
    


    double t_min = 1E9;
    double t = 0;


    int trial;
    int iter;
    for(trial=0;trial<num_trials;trial++){
	t = 0;

	t -= get_time();
	for(iter=0;iter<max_iter;iter++)
	    matrix_multiply(A,B,C,mkl_threads);
	t += get_time();

	printf("%6i: %9.6f\n",trial,t);

	if(t < t_min)
	    t_min = t;
    }
    printf("t_min: %9.6f\n",t_min);

    destroy_matrix(&A);
    destroy_matrix(&B);
    destroy_matrix(&C);

    return 0;
}

double get_time(){
    //output time in microseconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}





