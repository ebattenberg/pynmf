
#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"
#include<time.h>
#include<sys/time.h>


#define ITER_CHECK 25
#define MAX_ITER 200
#define CONVERGE_THRESH 0.008


void update_div(matrix W, matrix H, const matrix X);
double get_time();

/*void matrix_check(matrix A,int iter){
    int i;
    const int N = A.dim[0]*A.dim[1];
    for(i=0;i<N;i++){
	if(isnan(A.mat[i])){
	    printf("nan: %g, [%ix%i], %i\n",A.mat[i],A.dim[0],A.dim[1],iter);

	    exit(1);
	}
	if(0>(A.mat[i])){
	    printf("<0: %g, [%ix%i], %i\n",A.mat[i],A.dim[0],A.dim[1],iter);

	    exit(1);
	}
    }
}*/

/*void here(char* string){
    printf("\nhere: %s\n",string);
}*/


int main(){

    //factor X into W*H
    matrix W,H,X;
    
    read_matrix(&W,"../W.bin");
    read_matrix(&X,"../X.bin");
    read_matrix(&H,"../H.bin");

    update_div(W,H,X);

    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);

    return 0;
}

double get_time(){
    //output time in microseconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

void update_div(matrix W, matrix H, const matrix X){
    //run iterative multiplicative updates on W,H


    //initialize temp matrices -----------------------
    //
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
    
    
    double ttotal = 0;
    const int TIMERS = 9;
    double t[TIMERS];
    int i;
    for(i=0;i<TIMERS;i++)
	t[i] = 0.0;
    ttotal -= get_time();

    float diff,div,prev_div,change;
    matrix_multiply(W,H,WH);
    diff = matrix_difference_norm(X,WH);
    prev_div = matrix_div(X,WH);
    div = prev_div;
    printf("i: %4i, error: %6.4f, div: %8.4e\n",0,diff,prev_div);

    for(i=0;i<MAX_ITER;i++){

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
	    if(change < CONVERGE_THRESH){
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

	t[0] -= get_time();
	//WH = W*H
	matrix_multiply(W,H,WH);
	t[0] += get_time();

	t[1] -= get_time();
	//WH = WH+EPS
	matrix_eps(WH);
	t[1] += get_time();

	t[2] -= get_time();
	//Z = X./WH
	element_divide(X,WH,Z);
	t[2] += get_time();


	t[5] -= get_time();
	//sum cols of W into row vector
	sum_cols(W,sumW);
	t[5] += get_time();

	//convert sumW to col vector
	sumW.dim[0] = sumW.dim[1];
	sumW.dim[1] = 1;

	t[0] -= get_time();
	//WtZ = W'*Z
	matrix_multiply_AtB(W,Z,WtZ);
	t[0] += get_time();

	t[6] -= get_time();
	//WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
	//[element divide cols of WtZ by sumW']
	col_divide(WtZ,sumW,WtZ);
	t[6] += get_time();

	t[3] -= get_time();
	//H = H.*WtZ
	element_multiply(H,WtZ,H);
	t[3] += get_time();
	
	
	//
	// UPDATE W ---------------------------
	//

	t[0] -= get_time();
	//WH = W*H
	matrix_multiply(W,H,WH);
	t[0] += get_time();

	t[1] -= get_time();
	//WH = WH+EPS
	matrix_eps(WH);
	t[1] += get_time();

	t[2] -= get_time();
	//Z = X./WH
	element_divide(X,WH,Z);
	t[2] += get_time();

	t[4] -= get_time();
	//sum rows of H into col vector
	sum_rows(H,sumH2);
	t[4] += get_time();

	//convert sumH2 to row vector
	sumH2.dim[1] = sumH2.dim[0];
	sumH2.dim[0] = 1;

	t[0] -= get_time();
	//ZHt = Z*H'
	matrix_multiply_ABt(Z,H,ZHt);
	t[0] += get_time();

	t[7] -= get_time();
	//ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
	//[element divide rows of ZHt by sumH2']
	row_divide(ZHt,sumH2,ZHt);
	t[7] += get_time();

	t[3] -= get_time();
	//W = W.*ZHt
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

    ttotal += get_time();


    matrix_multiply(W,H,WH);
    diff = matrix_difference_norm(X,WH);
    prev_div = div;
    div = matrix_div(X,WH);
    change = (prev_div-div)/prev_div;
    printf("i: %4i, error: %6.4f, div: %8.4e, change: %8.5f\n",
	    i,diff,div,change);


    printf("t: %g\n",ttotal);
    for(i=0;i<TIMERS;i++)
	printf("t[%i]: %4.3e (%6.3f %%)\n",i,t[i],t[i]/ttotal*100);


    //free temporary matrices
    destroy_matrix(&WH);
    destroy_matrix(&Z);
    destroy_matrix(&WtZ);
    destroy_matrix(&ZHt);
    destroy_matrix(&sumH2);
    destroy_matrix(&sumW);

}





