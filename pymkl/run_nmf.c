

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<string.h>

#include<omp.h>
#include<mkl.h>

#include"../mkl/matrix.h"

//for FTZ/DAZ macros
#include <xmmintrin.h>
#include <pmmintrin.h>

#define ITER_CHECK 25
#define MAX_ITER 200
#define CONVERGE_THRESH 0
#define TIMERS 14


char *tname[] = {"total","sgemm","eps","vecdiv","vecmult","sumrows","sumcols","coldiv","rowdiv","check","sgemm1","sgemm2","sgemm3","sgemm4"};


int run_nmf(matrix X, matrix W, matrix H, int threads, int max_iter, int verbose);
void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double* t, int verbose);
double get_time(void);

int omp_threads;
int mkl_threads;
int eps_threads;
int vecdiv_threads;
int vecmult_threads;
int sumrows_threads;
int sumcols_threads;
int coldiv_threads;
int rowdiv_threads;
int check_threads;




int run_nmf(matrix X, matrix W, matrix H, int threads, int max_iter, int verbose)
{ 

    if (threads == 0 || threads > omp_get_max_threads())
    {
	omp_threads = omp_get_max_threads();
	mkl_threads = mkl_get_max_threads();
    }
    else
    {
	omp_threads = threads;
	mkl_threads = threads;
    }

    eps_threads = omp_threads;
    vecdiv_threads = omp_threads;
    vecmult_threads = omp_threads;
    sumrows_threads = omp_threads;
    sumcols_threads = omp_threads;
    coldiv_threads = omp_threads;
    rowdiv_threads = omp_threads;
    check_threads = omp_threads;


    double timers[TIMERS];


    int i;
    for(i=0;i<TIMERS;i++)
	timers[i]=0;


    update_div(W,H,X,CONVERGE_THRESH,max_iter,timers,verbose);


    return 0;
}

double get_time(){
    //output time in microseconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double *t,int verbose){
    //run iterative multiplicative updates on W,H


    //initialize temp matrices -----------------------
    //matrix to hold W*H
    matrix WH;
    create_matrix(&WH, W.dim[0], H.dim[1], 0.0);

    //matrix to hold X./(W*H+EPS)
    matrix Z;
    create_matrix(&Z, X.dim[0], X.dim[1], 0.0);

    //matrix to hold W'*Z
    matrix WtZ;
    create_matrix(&WtZ, W.dim[1], Z.dim[1], 0.0);

    //matrix to hold Z*H'
    matrix ZHt;
    create_matrix(&ZHt, Z.dim[0], H.dim[0], 0.0);

    //matrix to hold sum(W) [sum cols of W]
    matrix sumW;
    create_matrix(&sumW, 1, W.dim[1] ,0.0);

    //matrix to hold sum(H,2) [sum rows of H]
    matrix sumH2;
    create_matrix(&sumH2, H.dim[0], 1, 0.0);
    
    int i;
    
    if(t==NULL){
	double t_array[TIMERS];
	t = t_array;
	for(i=0;i<TIMERS;i++)
	    t[i] = 0;
    }

    //turn on the FTZ(15) and DAZ(6) bits in the floating point control register
    //FTZ = flush-to-zero, DAZ = denormal-as-zero
    //without these, sgemms slow down significantly as values approach zero
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    // the following does the same thing (by Waterman)
    /*
    unsigned int mxcsr;
    __asm__ __volatile__ ("stmxcsr (%0)" : : "r"(&mxcsr) : "memory");
    //mxcsr = (mxcsr | (1<<15) | (1<<6)) & ~((1<<11) | (1<<8));
    mxcsr = (mxcsr | (1<<15) | (1<<6)); 
    __asm__ __volatile__ ("ldmxcsr (%0)" : : "r"(&mxcsr));
    */


     

    float diff,div,prev_div,change;
    matrix_multiply(W,H,WH,mkl_threads);
    diff = matrix_difference_norm(X,WH, check_threads);
    prev_div = matrix_div(X,WH,check_threads);
    div = prev_div;
    if(verbose)
    {
	printf("OpenMP threads: %i\n",omp_threads);
	printf("i: %4i, error: %6.4f, div: %8.4e\n",0,diff,prev_div);
    }

    t[0] -= get_time();
    for(i=0;i<max_iter;i++){

	//check for convergence, print status
	if(i % ITER_CHECK == 0 && i != 0){
	    double tt = get_time();
	    matrix_multiply(W,H,WH,mkl_threads);
	    diff = matrix_difference_norm(X,WH,check_threads);
	    prev_div = div;
	    div = matrix_div(X,WH,check_threads);
	    change = (prev_div-div)/prev_div;
	    if(verbose)
		printf("i: %4i, error: %6.4f, div: %8.4e, change: %8.5f\n",
			i,diff,div,change);
	    if(change < thresh){
		printf("converged\n");
		break;
	    }
	    tt = get_time()-tt;
	    t[9] += tt;
	}
	    

	/* matlab algorithm
	   Z = X./(W*H+eps);
	   H = H.*(W'*Z)./(repmat(sum(W)',1,F));

	   Z = X./(W*H+eps);
	   W = W.*(Z*H')./(repmat(sum(H,2)',N,1));
	   */
		
	//
	// UPDATE H -----------------------------
	//

	//WH = W*H
	t[1] -= get_time();
	t[10] -= get_time();
	//matrix_eps(W,eps_threads);
	//matrix_eps(H,eps_threads);
	matrix_multiply(W,H,WH,mkl_threads);
	t[1] += get_time();
	t[10] += get_time();

	//WH = WH+EPS
	t[2] -= get_time();
	matrix_eps(WH,eps_threads);
	t[2] += get_time();

	//Z = X./WH
	t[3] -= get_time();
	element_divide(X,WH,Z,vecdiv_threads);
	t[3] += get_time();


	//sum cols of W into row vector
	t[6] -= get_time();
	sum_cols(W,sumW,sumcols_threads);
	t[6] += get_time();

	//convert sumW to col vector
	sumW.dim[0] = sumW.dim[1];
	sumW.dim[1] = 1;

	//WtZ = W'*Z
	t[1] -= get_time();
	t[11] -= get_time();
	matrix_multiply_AtB(W,Z,WtZ,mkl_threads);
	t[1] += get_time();
	t[11] += get_time();

	//WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
	//[element divide cols of WtZ by sumW']
	t[7] -= get_time();
	col_divide(WtZ,sumW,WtZ,coldiv_threads);
	t[7] += get_time();

	//H = H.*WtZ
	t[4] -= get_time();
	element_multiply(H,WtZ,H,vecmult_threads);
	t[4] += get_time();
	
	
	//
	// UPDATE W ---------------------------
	//

	//WH = W*H
	t[1] -= get_time();
	t[12] -= get_time();
	matrix_multiply(W,H,WH,mkl_threads);
	t[1] += get_time();
	t[12] += get_time();

	//WH = WH+EPS
	t[2] -= get_time();
	matrix_eps(WH,eps_threads);
	t[2] += get_time();

	//Z = X./WH
	t[3] -= get_time();
	element_divide(X,WH,Z,vecdiv_threads);
	t[3] += get_time();

	//sum rows of H into col vector
	t[5] -= get_time();
	sum_rows(H,sumH2,sumrows_threads);
	t[5] += get_time();

	//convert sumH2 to row vector
	sumH2.dim[1] = sumH2.dim[0];
	sumH2.dim[0] = 1;

	//ZHt = Z*H'
	t[1] -= get_time();
	t[13] -= get_time();
	matrix_multiply_ABt(Z,H,ZHt,mkl_threads);
	t[1] += get_time();
	t[13] += get_time();

	//ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
	//[element divide rows of ZHt by sumH2']
	t[8] -= get_time();
	row_divide(ZHt,sumH2,ZHt,rowdiv_threads);
	t[8] += get_time();

	//W = W.*ZHt
	t[4] -= get_time();
	element_multiply(W,ZHt,W,vecmult_threads);
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


    matrix_multiply(W,H,WH,mkl_threads);
    diff = matrix_difference_norm(X,WH,check_threads);
    prev_div = div;
    div = matrix_div(X,WH,check_threads);
    change = (prev_div-div)/prev_div;
    if(verbose){
	printf("i: %4i, error: %6.4f, div: %8.4e, change: %8.5f\n",
		i,diff,div,change);


	printf("\n");
	for(i=0;i<TIMERS;i++)
	    printf("t[%i]: %8.3f (%6.2f %%) %s\n",i,t[i],t[i]/t[0]*100,tname[i]);
    }


    //free temporary matrices
    destroy_matrix(&WH);
    destroy_matrix(&Z);
    destroy_matrix(&WtZ);
    destroy_matrix(&ZHt);
    destroy_matrix(&sumW);
    destroy_matrix(&sumH2);


}


