
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


double get_time();
void start_time(struct timeval *t);
double stop_time(struct timeval t0);

int omp_threads;

int main(int argc, char* argv[]){


    omp_threads = omp_get_max_threads();
    if (argc>1){
	if (atoi(argv[1]) < omp_threads)
	    omp_threads = atoi(argv[1]);
    }


    printf("omp_threads: \t\t%i\n",omp_threads);


    const int N = 512*3400;
    const int num_trials = 10;
    const long int ITER = 10;
    
    float *a, *b, *c;
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c = (float*)malloc(N*sizeof(float));

    int i;
    for(i=0;i<N;i++)
    {
	a[i] = (float)drand48();
	b[i] = (float)drand48();
    }
    
    


    double t;
    double t0;
    struct timeval ts;
    double min_t = 1E6;

    long int iter;
    int trial;
    for(trial=0;trial<num_trials;trial++){
	t = 0;


	for(iter=0;iter<ITER;iter++){
	t -= get_time();
	//printf("%i, ",iter);
	//start_time(&ts);
	//#pragma omp parallel for num_threads(1) schedule(static,512)
//#pragma omp parallel default(none) num_threads(omp_threads) shared(a,b,c,N) private(i)
	//{
	  //#pragma omp for schedule(static) nowait
#pragma omp parallel for num_threads(omp_threads)
	    for(i=0;i<N;i++){
		c[i] = a[i] / b[i];
	    }
	
	//}
	//t0 += stop_time(ts);
	t += get_time();
	}


	printf("trial: %i of %i: ",trial+1,num_trials);
	printf("%e sec\n",t);
	min_t = (min_t > t) ? t : min_t;
    }

    printf("\nmin = %g\n",min_t);




    free(a);
    free(b);
    free(c);





    return 0;
}

double get_time(){
    //output time in microseconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

void start_time(struct timeval *t)
{
    gettimeofday(t,NULL);
}

double stop_time(struct timeval t0)
{
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)((t.tv_sec-t0.tv_sec) + (t.tv_usec-t0.tv_usec)/1E6);
}

