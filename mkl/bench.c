
#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"
#include<time.h>
#include<sys/time.h>
#include<string.h>

#include<omp.h>
#include<mkl.h>


#define ITER_CHECK 25
#define MAX_ITER 100
#define CONVERGE_THRESH 0
#define TIMERS 10
#define TRIALS 10
#define OMP_THREADS 2
#define MKL_THREADS 2


char *tname[] = {"total","sgemm","eps","vecdiv","vecmult","sumrows","sumcols","coldiv","rowdiv","check"};


void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double* t, int verbose);
double get_time();

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
    if(argc > 1){
	if(!strcmp(argv[1],"-h")){
	    printf("usage: bench iterations[100] trials[10] omp_threads[#procs] mkl_threads[#procs] eps_threads vecdiv_threads vecmult_threads sumrows_threads sumcols_threads coldiv_threads rowdiv_threads check_threads\n");
	    exit(0);	
	}
	else
	    max_iter = atoi(argv[1]);
    }
    else 
	max_iter = MAX_ITER;

    int num_trials;
    if(argc>2)
	num_trials = atoi(argv[2]);
    else 
	num_trials = TRIALS;

    int verbose = 0;
    if (num_trials<10)
	verbose = 1;

    omp_threads = omp_get_max_threads();
    if (argc>3){
	if (atoi(argv[3]) < omp_threads)
	    omp_threads = atoi(argv[3]);
    }

    mkl_threads = mkl_get_max_threads();
    if (argc>4){
	if (atoi(argv[4]) < mkl_threads)
	    mkl_threads = atoi(argv[4]);
    }

    if (argc>5)
	eps_threads = atoi(argv[5]);
    else
	eps_threads = omp_threads;

    if (argc>6)
vecdiv_threads = atoi(argv[6]);
    else
	vecdiv_threads = omp_threads;

    if (argc>7)
	vecmult_threads = atoi(argv[7]);
    else
	vecmult_threads = omp_threads;

    if (argc>8)
	sumrows_threads = atoi(argv[8]);
    else
	sumrows_threads = omp_threads;

    if (argc>9)
	sumcols_threads = atoi(argv[9]);
    else
	sumcols_threads = omp_threads;

    if (argc>10)
	coldiv_threads = atoi(argv[10]);
    else
	coldiv_threads = omp_threads;

    if (argc>11)
	rowdiv_threads = atoi(argv[11]);
    else
	rowdiv_threads = omp_threads;

    if (argc>12)
	check_threads = atoi(argv[12]);
    else
	check_threads = omp_threads;

    printf("omp_threads: \t\t%i\n",omp_threads);
    printf("mkl_threads: \t\t%i\n",mkl_threads);
    printf("eps_threads: \t\t%i\n", eps_threads);
    printf("vecdiv_threads: \t%i\n", vecdiv_threads);
    printf("vecmult_threads: \t%i\n", vecmult_threads);
    printf("sumrows_threads: \t%i\n", sumrows_threads);
    printf("sumcols_threads: \t%i\n", sumcols_threads);
    printf("coldiv_threads: \t%i\n", coldiv_threads);
    printf("rowdiv_threads: \t%i\n", rowdiv_threads);
    printf("check_threads: \t\t%i\n", check_threads);


    
    //get cpuinfo
    char cpuinfo[128];
    FILE* fp = popen("/usr/bin/less /proc/cpuinfo | grep \"model name\"","r");
    fgets(cpuinfo,128,fp);
    pclose(fp);

    char filename[128];
    sprintf(filename,"./results/bench_%i_%i_%i_%i_%i_%i_%i_%i_%i_%i_%i.out",max_iter,omp_threads,mkl_threads,eps_threads,
	    vecdiv_threads, vecmult_threads, sumrows_threads, sumcols_threads, coldiv_threads, rowdiv_threads, check_threads);
    fp = fopen(filename,"w");
    if ( fp == NULL ){
	fprintf(stderr,"error: unsuccessful opening output file\n");
	exit(1);
    }
    fprintf(fp,"%s\n",cpuinfo);
    fprintf(fp,"omp_threads: %i\nmkl_threads: %i\niterations: %i\n\n",omp_threads,mkl_threads,max_iter);
    fprintf(fp,"trial\t\t");
    int i;
    for(i=0;i<TIMERS;i++)
	fprintf(fp,"%s\t\t",tname[i]);
    fprintf(fp,"diff(W)\t\tdiff(H)");
    fprintf(fp,"\n");


    double timers[num_trials][TIMERS];
    float diff[num_trials][2];


    int trial;
    for(trial=0;trial<num_trials;trial++){
	for(i=0;i<TIMERS;i++)
	    timers[trial][i]=0;
	copy_matrix(W0,W);
	copy_matrix(H0,H);

	update_div(W,H,X,CONVERGE_THRESH,max_iter,timers[trial],verbose);

	diff[trial][0] = matrix_difference_norm(Wf,W,check_threads);
	diff[trial][1] = matrix_difference_norm(Hf,H,check_threads);
	fprintf(fp,"%i\t\t",trial);
	for(i=0;i<TIMERS;i++)
	    fprintf(fp,"%6.3e\t",timers[trial][i]);
	fprintf(fp,"%6.3e\t%6.3e\n",diff[trial][0],diff[trial][1]);
	printf("trial: %i of %i: ",trial+1,num_trials);
	printf("%g sec\n",timers[trial][0]);
    }








    fclose(fp);

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

    float diff,div,prev_div,change;
    matrix_multiply(W,H,WH,mkl_threads);
    diff = matrix_difference_norm(X,WH, check_threads);
    prev_div = matrix_div(X,WH,check_threads);
    div = prev_div;
    if(verbose)
	printf("i: %4i, error: %6.4f, div: %8.4e\n",0,diff,prev_div);

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
		//break;
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
	matrix_multiply(W,H,WH,mkl_threads);
	t[1] += get_time();

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
	matrix_multiply_AtB(W,Z,WtZ,mkl_threads);
	t[1] += get_time();

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
	matrix_multiply(W,H,WH,mkl_threads);
	t[1] += get_time();

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
	matrix_multiply_ABt(Z,H,ZHt,mkl_threads);
	t[1] += get_time();

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




