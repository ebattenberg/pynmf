
#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"
#include<time.h>
#include<sys/time.h>

#include<omp.h>
#include<mkl.h>


#define ITER_CHECK 25
#define MAX_ITER 200
#define CONVERGE_THRESH 0
#define TIMERS 10
#define TRIALS 2

#ifndef OMP_THREADS
#define OMP_THREADS 2
#endif

#ifndef MKL_THREADS
#define MKL_THREADS 2
#endif

int mkl_threads = MKL_THREADS;
int omp_threads = OMP_THREADS;

char *tname[] = {"sgemm","eps","vecdiv","vecmult","sumrows","sumcols","coldiv","rowdiv","check","total"};


void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double* t);
double get_time();




int main(int argc, char* argv[]){


    //factor X into W*H
    matrix W,H,X,W0,H0,Wf,Hf;
    
    read_matrix(&W0,"../W.bin");
    read_matrix(&H0,"../H.bin");
    read_matrix(&X,"../X.bin");

    //final matrices from matlab version (after 200 iterations)
    read_matrix(&Wf,"../Wf.bin");
    read_matrix(&Hf,"../Hf.bin");

    create_matrix(&W,W0.dim[0],W0.dim[1],0);
    create_matrix(&H,H0.dim[0],H0.dim[1],0);

    int max_iter;
    if(argc > 1)
	max_iter = atoi(argv[1]);
    else 
	max_iter = MAX_ITER;

    if (argc>2)
	omp_threads = atoi(argv[2]);
    else
	omp_threads = omp_get_max_threads();

    if (argc>3)
	mkl_threads = atoi(argv[3]);
    else
	mkl_threads = mkl_get_max_threads();

    printf("omp threads: %i, mkl threads: %i\n",omp_threads,mkl_threads);

    


    double timers[TIMERS] = {0};
    float diff[2];

    copy_matrix(W0,W);
    copy_matrix(H0,H);
    update_div(W,H,X,CONVERGE_THRESH,max_iter,timers);
    diff[0] = matrix_difference_norm(Wf,W);
    diff[1] = matrix_difference_norm(Hf,H);




    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);
    destroy_matrix(&W0);
    destroy_matrix(&H0);
    destroy_matrix(&Wf);
    destroy_matrix(&Hf);

    return 0;
}

double get_time(){
    //output time in microseconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double *t){
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

    float diff,div,prev_div,change;
    matrix_multiply(W,H,WH);
    diff = matrix_difference_norm(X,WH);
    prev_div = matrix_div(X,WH);
    div = prev_div;
    printf("i: %4i, error: %6.4f, div: %8.4e\n",0,diff,prev_div);

    t[9] -= get_time();
    for(i=0;i<max_iter;i++){

	//check for convergence, print status
	if(i % ITER_CHECK == 0 && i != 0){
	    double tt = get_time();
	    matrix_multiply(W,H,WH);
	    diff = matrix_difference_norm(X,WH);
	    prev_div = div;
	    div = matrix_div(X,WH);
	    change = (prev_div-div)/prev_div;
	    printf("i: %4i, error: %6.4f, div: %8.4e, change: %8.5f\n",
		    i,diff,div,change);
	    if(change < thresh){
		printf("converged\n");
		//break;
	    }
	    tt = get_time()-tt;
	    t[8] += tt;
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
	t[0] -= get_time();
	matrix_multiply(W,H,WH);
	t[0] += get_time();

	//WH = WH+EPS
	t[1] -= get_time();
	matrix_eps(WH);
	t[1] += get_time();

	//Z = X./WH
	t[2] -= get_time();
	element_divide(X,WH,Z);
	t[2] += get_time();


	//sum cols of W into row vector
	t[5] -= get_time();
	sum_cols(W,sumW);
	t[5] += get_time();

	//convert sumW to col vector
	sumW.dim[0] = sumW.dim[1];
	sumW.dim[1] = 1;

	//WtZ = W'*Z
	t[0] -= get_time();
	matrix_multiply_AtB(W,Z,WtZ);
	t[0] += get_time();

	//WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
	//[element divide cols of WtZ by sumW']
	t[6] -= get_time();
	col_divide(WtZ,sumW,WtZ);
	t[6] += get_time();

	//H = H.*WtZ
	t[3] -= get_time();
	element_multiply(H,WtZ,H);
	t[3] += get_time();
	
	
	//
	// UPDATE W ---------------------------
	//

	//WH = W*H
	t[0] -= get_time();
	matrix_multiply(W,H,WH);
	t[0] += get_time();

	//WH = WH+EPS
	t[1] -= get_time();
	matrix_eps(WH);
	t[1] += get_time();

	//Z = X./WH
	t[2] -= get_time();
	element_divide(X,WH,Z);
	t[2] += get_time();

	//sum rows of H into col vector
	t[4] -= get_time();
	sum_rows(H,sumH2);
	t[4] += get_time();

	//convert sumH2 to row vector
	sumH2.dim[1] = sumH2.dim[0];
	sumH2.dim[0] = 1;

	//ZHt = Z*H'
	t[0] -= get_time();
	matrix_multiply_ABt(Z,H,ZHt);
	t[0] += get_time();

	//ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
	//[element divide rows of ZHt by sumH2']
	t[7] -= get_time();
	row_divide(ZHt,sumH2,ZHt);
	t[7] += get_time();

	//W = W.*ZHt
	t[3] -= get_time();
	element_multiply(W,ZHt,W);
	t[3] += get_time();



	// ------------------------------------

	//reset sumW to row vector
	sumW.dim[1] = sumW.dim[0];
	sumW.dim[0] = 1;
	//reset sumH2 to col vector
	sumH2.dim[0] = sumH2.dim[1];
	sumH2.dim[1] = 1;

	// ---------------------------------------
	
	    
    }

    t[9] += get_time();


    matrix_multiply(W,H,WH);
    diff = matrix_difference_norm(X,WH);
    prev_div = div;
    div = matrix_div(X,WH);
    change = (prev_div-div)/prev_div;
    printf("i: %4i, error: %6.4f, div: %8.4e, change: %8.5f\n",
	    i,diff,div,change);


    printf("\n");
    for(i=0;i<TIMERS-1;i++)
	printf("t[%i]: %8.3f (%6.2f %%) %s\n",i,t[i],t[i]/t[TIMERS-1]*100,tname[i]);
    printf("________________________________________\n");
    printf("t[%i]: %8.3f (%6.2f %%) %s\n",i,t[i],t[i]/t[TIMERS-1]*100,tname[i]);


    //free temporary matrices
    destroy_matrix(&WH);
    destroy_matrix(&Z);
    destroy_matrix(&WtZ);
    destroy_matrix(&ZHt);
    destroy_matrix(&sumH2);
    destroy_matrix(&sumW);

}





