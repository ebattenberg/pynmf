#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"
#include<time.h>
#include<sys/time.h>
#include<string.h>

#define MAX_ITER 200
#define TRIALS 10
#define MAT_DIM 100




double get_time();

int main(int argc, char* argv[]){




    if(argc > 1){
	if(!strcmp(argv[1],"-h")){
	    printf("usage: bench matrix_dim1[100] matrix_dim2[100] matrix_dim3[100] iterations[100] trials[10]\n");
	    exit(0);	
	}
    }

    int matrix_dim[3];
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

    int max_iter = MAX_ITER;
    if(argc>4)
	max_iter = atoi(argv[4]);

    int num_trials = TRIALS;
    if(argc>5)
	num_trials = atoi(argv[5]);


    matrix A,B,C;

    create_matrix_on_both(&A,matrix_dim[0],matrix_dim[1],0);
    create_matrix_on_both(&B,matrix_dim[1],matrix_dim[2],0);
    create_matrix_on_both(&C,matrix_dim[0],matrix_dim[2],0);


    double t;
    double t_min = 1E9;

    int trial;
    int iter;
    for(trial=0;trial<num_trials;trial++){
	t = 0;

	t -= get_time();
	for(iter=0;iter<max_iter;iter++)
	    matrix_multiply_d(A,B,C);
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
    cudaThreadSynchronize();
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

