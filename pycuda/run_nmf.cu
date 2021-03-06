#include<stdio.h>
#include<stdlib.h>
#include"../cuda/matrix.h"
#include<time.h>
#include<sys/time.h>
#include<string.h>

#define ITER_CHECK 25
#define MAX_ITER 200
#define CONVERGE_THRESH 0
#define TIMERS 10
#define TRIALS 10


char *tname[] = {"total","sgemm","eps","vecdiv","vecmult","sumrows","sumcols","coldiv","rowdiv","check"};

int vecdiv_block_size = BLOCK_SIZE;
int eps_block_size = BLOCK_SIZE;


void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double* t, int verbose);
double get_time();

int run_nmf(matrix X, matrix W, matrix H, int max_iter, int verbose)
{

    cudaSetDevice(0);


    double timers[TIMERS];

    int i;
    for(i=0;i<TIMERS;i++)
	timers[i]=0;




    copy_matrix_to_device(&X);
    printf("here\n");
    copy_matrix_to_device(&W);
    copy_matrix_to_device(&H);

    update_div(W,H,X,CONVERGE_THRESH,max_iter,timers,verbose);

    copy_matrix_from_device(&W);
    copy_matrix_from_device(&H);


    cudaFree(W.mat_d);
    cudaFree(H.mat_d);
    cudaFree(X.mat_d);
    W.mat_d = NULL;
    H.mat_d = NULL;
    X.mat_d = NULL;


    return 0;
}

double get_time(){
    //output time in microseconds
    cudaThreadSynchronize();
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

void update_div(matrix W0, matrix H0, matrix X0, const float thresh, const int max_iter, double *t,int verbose){
    //run iterative multiplicative updates on W,H

    cublasInit();

    const int M = W0.dim[0];
    const int K = W0.dim[1];
    const int N = H0.dim[1];

    const int PAD_MULT = 32;

    int M_padded = M;
    if (M%PAD_MULT != 0)
	M_padded = M + (PAD_MULT - (M % PAD_MULT));

    int K_padded = K;
    if (K%PAD_MULT != 0){
	K_padded = K + (PAD_MULT - (K % PAD_MULT));
    }

    int N_padded = N;
    if (N%PAD_MULT != 0)
	N_padded = N + (PAD_MULT - (N % PAD_MULT));

    //initialize temp matrices -----------------------
    //matrix to hold W*H
    matrix WH0;
    create_matrix_on_device(&WH0,M,N,0.0);

    //matrix to hold X./(W*H+EPS)
    matrix Z;
    create_matrix_on_device(&Z,M_padded,N_padded,0.0);

    //matrix to hold W'*Z
    matrix WtZ;
    create_matrix_on_device(&WtZ,K_padded,N_padded,0.0);

    //matrix to hold Z*H'
    matrix ZHt;
    create_matrix_on_device(&ZHt,M_padded,K_padded,0.0);

    //matrix to hold sum(W) [sum of cols of W]
    matrix sumW;
    create_matrix_on_device(&sumW,1,K_padded,0.0);

    //matrix to hold sum(H,2) [sum of rows of H]
    matrix sumH2;
    create_matrix_on_device(&sumH2,K_padded,1,0.0);


    //printf("M K N = %i %i %i\n", M, K, N);
    //printf("M K N = %i %i %i\n", M_padded, K_padded, N_padded);


    //matrices to hold padded versions of matrices
    matrix W;
    create_matrix_on_device(&W,M_padded,K_padded,0.0);

    matrix H;
    create_matrix_on_device(&H,K_padded,N_padded,0.0);

    matrix X;
    create_matrix_on_device(&X,M_padded,N_padded,0.0);

    matrix WH;
    create_matrix_on_device(&WH,M_padded,N_padded,0.0);


    copy_to_padded(W0,W);
    copy_to_padded(H0,H);
    copy_to_padded(X0,X);


    if(t==NULL){
	double t_array[TIMERS];
	t = t_array;
	for(int i=0;i<TIMERS;i++)
	    t[i] = 0;
    }

    float diff,div,prev_div,change;
    copy_from_padded(W0,W);
    copy_from_padded(H0,H);
    matrix_multiply_d(W0,H0,WH0);
    diff = matrix_difference_norm_d(compute,X0,WH0,128,32,128,4);
    prev_div = matrix_div_d(compute,X0,WH0,128,32,128,4);
    div = prev_div;
    if(verbose)
	printf("i: %4i, error: %6.4f, div: %8.4e\n",0,diff,prev_div);

    t[0] -= get_time();

    int i;
    for(i=0;i<max_iter;i++){

	//check for convergence, print status
	t[9] -= get_time();
	if(i % ITER_CHECK == 0 && i != 0){
	    copy_from_padded(W0,W);
	    copy_from_padded(H0,H);
	    matrix_multiply_d(W0,H0,WH0);
	    diff = matrix_difference_norm_d(compute,X0,WH0,128,32,128,4);
	    prev_div = div;
	    div = matrix_div_d(compute,X0,WH0,128,32,128,4);
	    change = (prev_div-div)/prev_div;
	    if(verbose)
		printf("i: %4i, error: %6.4f, div: %8.4e, change: %8.5f\n",
			i,diff,div,change);
	    if(change < thresh){
		printf("converged\n");
		//break;
	    }
	}
	t[9] += get_time();



	/* matlab algorithm
	   Z = X./(W*H+eps); H = H.*(W'*Z)./(repmat(sum(W)',1,F)); 
	   Z = X./(W*H+eps);
	   W = W.*(Z*H')./(repmat(sum(H,2)',N,1));
	   */
		
	//
	// UPDATE H -----------------------------
	//


	//WH = W*H
	t[1] -= get_time();
	matrix_multiply_d(W,H,WH);
	t[1] += get_time();

	//WH = WH+EPS
	t[2] -= get_time();
	matrix_eps_d(WH,eps_block_size);
	t[2] += get_time();

	//Z = X./WH
	t[3] -= get_time();
	element_divide_d(X,WH,Z,vecdiv_block_size);
	t[3] += get_time();


	//sum cols of W into row vector
	t[6] -= get_time();
	sum_cols_d(compute,W,sumW,128,4,1,1);
	matrix_eps_d(sumW,32);
	t[6] += get_time();

	//convert sumW to col vector
	sumW.dim[0] = sumW.dim[1];
	sumW.dim[1] = 1;

	//WtZ = W'*Z
	t[1] -= get_time();
	matrix_multiply_AtB_d(W,Z,WtZ);
	t[1] += get_time();

	//WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
	//[element divide cols of WtZ by sumW']
	t[7] -= get_time();
	col_divide_d(WtZ,sumW,WtZ);
	t[7] += get_time();

	//H = H.*WtZ
	t[4] -= get_time();
	element_multiply_d(H,WtZ,H);
	t[4] += get_time();


	//
	// UPDATE W ---------------------------
	//

	//WH = W*H
	t[1] -= get_time();
	matrix_multiply_d(W,H,WH);
	t[1] += get_time();

	//WH = WH+EPS
	t[2] -= get_time();
	matrix_eps_d(WH,eps_block_size);
	t[2] += get_time();

	//Z = X./WH
	t[3] -= get_time();
	element_divide_d(X,WH,Z,vecdiv_block_size);
	t[3] += get_time();

	//sum rows of H into col vector
	t[5] -= get_time();
	sum_rows_d(compute,H,sumH2,128,32,1,1);
	matrix_eps_d(sumH2,32);
	t[5] += get_time();

	//convert sumH2 to row vector
	sumH2.dim[1] = sumH2.dim[0];
	sumH2.dim[0] = 1;

	//ZHt = Z*H'
	t[1] -= get_time();
	matrix_multiply_ABt_d(Z,H,ZHt);
	t[1] += get_time();

	//ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
	//[element divide rows of ZHt by sumH2']
	t[8] -= get_time();
	row_divide_d(ZHt,sumH2,ZHt);
	t[8] += get_time();

	//W = W.*ZHt
	t[4] -= get_time();
	element_multiply_d(W,ZHt,W);
	t[4] += get_time();



	// ------------------------------------

	//reset sumW to row vector
	sumW.dim[1] = sumW.dim[0];
	sumW.dim[0] = 1;
	//reset sumH2 to col vector
	sumH2.dim[0] = sumH2.dim[1];
	sumH2.dim[1] = 1;

	// ---------------------------------------

    }

    t[0] += get_time();



    copy_from_padded(W0,W);
    copy_from_padded(H0,H);
    matrix_multiply_d(W0,H0,WH0);
    diff = matrix_difference_norm_d(compute,X0,WH0,128,32,128,4);
    prev_div = div;
    div = matrix_div_d(compute,X0,WH0,128,32,128,4);
    change = (prev_div-div)/prev_div;
    if(verbose){
	printf("i: %4i, error: %6.4f, div: %8.4e, change: %8.5f\n",
		i,diff,div,change);

	printf("\n");
	for(i=0;i<TIMERS;i++)
	    printf("t[%i]: %8.3f (%6.2f %%) %s\n",i,t[i],t[i]/t[0]*100,tname[i]);
    }

    //clean up extra reduction memory
    matrix_difference_norm_d(cleanup,X,WH,0,0,0,0);
    matrix_div_d(cleanup,X,WH,0,0,0,0);
    sum_cols_d(cleanup,W,sumW,0,0,0,0);
    sum_rows_d(cleanup,H,sumH2,0,0,0,0);

    destroy_matrix(&WH);
    destroy_matrix(&WH0);
    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);
    destroy_matrix(&Z);
    destroy_matrix(&WtZ);
    destroy_matrix(&ZHt);
    destroy_matrix(&sumW);
    destroy_matrix(&sumH2);

    cublasShutdown();

}
