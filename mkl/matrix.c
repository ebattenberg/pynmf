#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
//#include "/usr/lib/gcc/x86_64-linux-gnu/4.3/include/omp.h"
#include <omp.h>
#include <xmmintrin.h>



void read_matrix(matrix* A, char* file){
    //read matrix in from file, store in column-major order
    //then copy matrix to device
    //A* must point to an uninitialized matrix

    FILE* fp;    
    size_t count;
    
    fp = fopen(file,"rb");
    count = fread(A->dim,sizeof(int),2, fp); 
    if(count < 2)
	fprintf(stderr,"read_matrix: fread error\n");

    int N = A->dim[0]*A->dim[1];
    A->mat = (float*)malloc(sizeof(float)*N);
    if((size_t)A->mat%16)
	printf("not 16 byte aligned\n");
    //A->mat = (float*)MKL_malloc(sizeof(float)*N,16);
    count = fread(A->mat,sizeof(float),N,fp);
    if(count < N)
	fprintf(stderr,"read_matrix: fread error\n");
    fclose(fp);

    printf("read %s [%ix%i]\n",file,A->dim[0],A->dim[1]);
}

void copy_matrix(matrix A, matrix B){
    //B = A;

    if(A.dim[0]!=B.dim[0] || A.dim[1]!=B.dim[1]){
	fprintf(stderr,"copy_matrix: dimension error\n");
	exit(1);
    }
    const int N = A.dim[0]*A.dim[1];
    int i;
    for(i=0;i<N;i++)
	B.mat[i] = A.mat[i];

}

void write_matrix(matrix A, char* file){
    //write matrix to file using column-major order
    //dimensions are written as leading ints

    FILE* fp;    
    size_t count;
    
    fp = fopen(file,"wb");
    count = fwrite(A.dim,sizeof(int),2,fp); 
    if(count < 2)
	fprintf(stderr,"write_matrix: fwrite error\n");

    
    count = fwrite(A.mat,sizeof(float),A.dim[0]*A.dim[1],fp);
    if(count < A.dim[0]*A.dim[1])
	fprintf(stderr,"write_matrix: fwrite error\n");
    fclose(fp);

    printf("write %s [%ix%i]\n",file,A.dim[0],A.dim[1]); 
}

void create_matrix(matrix* A, int rows, int cols, float value){
    //create matrix with all elements equal to 'value'
    //matrix dimensions are in dim (rows,cols)

    A->dim[0] = rows;
    A->dim[1] = cols;

    //A->mat = (float*)MKL_malloc(sizeof(float)*A->dim[0]*A->dim[1],16);
    A->mat = (float*)malloc(sizeof(float)*A->dim[0]*A->dim[1]);

    int i;
    for(i=0;i<A->dim[0]*A->dim[1];i++)
	A->mat[i] = value;
}

void destroy_matrix(matrix* A){
    //create matrix with all elements equal to 'value'
    //matrix dimensions are in dim (rows,cols)

    if(A->mat != NULL)
	free(A->mat);
	//MKL_free(A->mat);
    A->mat = NULL;
    A->dim[0] = 0;
    A->dim[1] = 0;

}

void print_matrix(matrix A){
    int i,j;
    printf("\n");
    const int lda = A.dim[0];
    const int tda = A.dim[1];
    for(i=0;i<lda;i++){
	for(j=0;j<tda;j++){
	    printf("% 5.5g ",A.mat[i+A.dim[0]*j]);
	}
	printf("\n");
    }
    printf("\n");
}

void naive_matrix_multiply( matrix a, matrix b, matrix c) {
    // c[m][p] = a[m][n] times b[n][p]
    int i,j,k;
    int m = a.dim[0];
    int n = b.dim[0];
    int p = b.dim[1];
    for (j = 0; j < p; j++) {
	for (i = 0; i < m; i++) {
	    float s = 0;
	    for (k = 0; k < n; k++) {
		s += a.mat[i + m*k] * b.mat[k + n*j];
	    }
	    c.mat[i + m*j] = s;
	}
    }
}

void matrix_transpose( matrix a, matrix b) {
    if(a.dim[0] != b.dim[1] || a.dim[1] != b.dim[0]){
	fprintf(stderr,"matrix_transpose: dimension error\n");
	exit(1);
    }
    int i,j;
    const int m = a.dim[0];
    const int n = b.dim[0];
    const int p = a.dim[1];
    for (j = 0; j < p; j++) 
	for (i = 0; i < m; i++) {
	    b.mat[j+i*n] = a.mat[j*m+i];
	}
    
}

void matrix_multiply( matrix a, matrix b, matrix c, int threads){

    //matrix_eps(a, threads);
    //matrix_eps(b, threads);

    mkl_set_num_threads(threads);
    cblas_sgemm(CblasColMajor, CblasNoTrans,
	    CblasNoTrans, c.dim[0], c.dim[1],
	    a.dim[1], 1, a.mat,
	    a.dim[0], b.mat, b.dim[0],
	    0, c.mat, c.dim[0]);
}

void matrix_multiply_AtB( matrix a, matrix b, matrix c, int threads){
    
    //matrix_eps(a, threads);
    //matrix_eps(b, threads);

    mkl_set_num_threads(threads);
    
    cblas_sgemm(CblasColMajor, CblasTrans,
	    CblasNoTrans, c.dim[0], c.dim[1],
	    b.dim[0], 1, a.mat,
	    a.dim[0], b.mat, b.dim[0],
	    0, c.mat, c.dim[0]);
}

void matrix_multiply_ABt( matrix a, matrix b, matrix c, int threads ){
    
    //matrix_eps(a, threads);
    //matrix_eps(b, threads);

    mkl_set_num_threads(threads);
    
    cblas_sgemm(CblasColMajor, CblasNoTrans,
	    CblasTrans, c.dim[0], c.dim[1],
	    a.dim[1], 1, a.mat,
	    a.dim[0], b.mat, b.dim[0],
	    0, c.mat, c.dim[0]);
}

float matrix_difference_norm(matrix A, matrix B, int omp_threads){
    int i;
    float s = 0;
    float m = 0;
    float x;
    const float* const a = A.mat;
    const float* const b = B.mat;
    if(A.dim[0] != B.dim[0] || A.dim[1] != B.dim[1]){
	fprintf(stderr,"matrix_difference_norm: dimensions do not agree\n");
	exit(1);
    }
    const int N = A.dim[0]*A.dim[1];

#pragma omp parallel for reduction(+:s,m) private(x) num_threads(omp_threads)
    for(i=0;i<N;i++){
	x = a[i];
	s += fabs(x - b[i]);
	m += x;
    }
    return s/m;
}

float matrix_difference_max(matrix A, matrix B){
    int i;
    int ind = 0;
    float s = 0;
    float x;
    const float* const a = A.mat;
    const float* const b = B.mat;
    if(A.dim[0] != B.dim[0] || A.dim[1] != B.dim[1]){
	fprintf(stderr,"matrix_difference_max: dimensions do not agree\n");
	exit(1);
    }
    const int N = A.dim[0]*A.dim[1];
    for(i=0;i<N;i++){
	x = (a[i]-b[i]);
	if (fabs(s)<fabs(x)){
	    s=x;
	    ind = i;
	}
    }
    printf("a[i]: %g, b[i]: %g\n",a[ind],b[ind]);
    return s;
}

float matrix_div(matrix A, matrix B, int omp_threads){
    int i;
    float s = 0;
    float x,y;
    const float* const a = A.mat;
    const float* const b = B.mat;
    if(A.dim[0] != B.dim[0] || A.dim[1] != B.dim[1]){
	fprintf(stderr,"matrix_difference_norm: dimensions do not agree\n");
	exit(1);
    }
    const int N = A.dim[0]*A.dim[1];

#pragma omp parallel for reduction(+:s) private(x,y) num_threads(omp_threads)
    for(i=0;i<N;i++){
	x = a[i];
	y = b[i];
	s += x*(log(x)-log(y))-x+y;
	//s += x*(log(x/y))-x+y;
	//if (x*(log(x/y))-x+y < 0)
	    //printf("x=%g y=%g\n",x,y);
    }
    return s;
}

void element_divide( matrix a, matrix b, matrix c, int omp_threads){
    // c = a./b
    int i;

    if(a.dim[0] != b.dim[0] || a.dim[0] != c.dim[0] ||
	    a.dim[1] != b.dim[1] || a.dim[1] != c.dim[1]){
	fprintf(stderr,"element_divide: dimensions do not agree\n");
	exit(1);
    }
    const int N = a.dim[0]*a.dim[1];
    const float* const am = a.mat;
    const float* const bm = b.mat;
    float* const cm = c.mat;


#pragma omp parallel for num_threads(omp_threads)
	for(i=0;i<N;i++){
	    cm[i] = am[i] / bm[i];
	}

}

void element_multiply( matrix a, matrix b, matrix c, int omp_threads){
    // c = a./b
    int i;

    if(a.dim[0] != b.dim[0] || a.dim[0] != c.dim[0] ||
	    a.dim[1] != b.dim[1] || a.dim[1] != c.dim[1]){
	fprintf(stderr,"element_multiply: dimensions do not agree\n");
	exit(1);
    }
    const int N = a.dim[0]*a.dim[1];

#pragma omp parallel for num_threads(omp_threads)
    for(i=0;i<N;i++)
	c.mat[i] = a.mat[i]*b.mat[i];
}

void matrix_add_constant( matrix a, const float c){
    // c = a./b
    int i;

    const int N = a.dim[0]*a.dim[1];
    for(i=0;i<N;i++)
	a.mat[i] += c;
}

void matrix_eps( matrix a, int omp_threads){
    // c = a./b
    int i;

    const int N = a.dim[0]*a.dim[1];

#pragma omp parallel for num_threads(omp_threads)
    for(i=0;i<N;i++)
	a.mat[i] += EPS;
	//if(a.mat[i] < EPS)
	    //a.mat[i] = EPS;
}

void row_divide( matrix a, matrix b, matrix c, int omp_threads){
    //element divide every row of 'a' by row vector 'b'

    if(a.dim[1] != b.dim[1] || a.dim[0] != c.dim[0] ||
	    a.dim[1] != c.dim[1] || b.dim[0] != 1)
    {
	fprintf(stderr,"row_divide: dimension error\n");
	exit(1);
    }

    int i,j;
    const int lda = a.dim[0];
    const int tda = a.dim[1];
    float* A = a.mat;
    float* B = b.mat;
    float* C = c.mat;
    int ldaj;
    float recip;

    float Brecip[tda];

#pragma omp parallel for num_threads(omp_threads)
    for(j=0;j<tda;j++)
	Brecip[j] = 1/B[j];

#pragma omp parallel for num_threads(omp_threads)
    for(j=0;j<tda;j++){
	recip = Brecip[j];
	ldaj = lda*j;
	for(i=0;i<lda;i++)
	    C[i+ldaj] = A[i+ldaj]*recip;
    }
}

void col_divide( matrix a, matrix b, matrix c, int omp_threads){
    //element divide every column of 'a' by column vector 'b'

    if(a.dim[0] != b.dim[0] || a.dim[0] != c.dim[0] ||
	    a.dim[1] != c.dim[1] || b.dim[1] != 1)
    {
	fprintf(stderr,"col_divide: dimension error\n");
	exit(1);
    }

    int i,j;
    const int lda = a.dim[0];
    const int tda = a.dim[1];
    float* A = a.mat;
    float* B = b.mat;
    float* C = c.mat;
    int ldaj;

    float Brecip[lda];

#pragma omp parallel for num_threads(omp_threads)
    for(i=0;i<lda;i++)
	Brecip[i] = 1/B[i];

#pragma omp parallel for num_threads(omp_threads)
    for(j=0;j<tda;j++){
	ldaj = lda*j;
	for(i=0;i<lda;i++){
	    C[i+ldaj] = A[i+ldaj]*Brecip[i];
	}
    }

}

void sum_rows( matrix a, matrix c, int omp_threads){
    int i,j;
    
    if(a.dim[0] == c.dim[0] && c.dim[1] == 1){

	const int lda = a.dim[0];
	const int tda = a.dim[1];

	for(i=0;i<lda;i++)
	    c.mat[i] = 0;
	
	for(j=0;j<tda;j++){
	    const int ldaj = lda*j;
	    for(i=0;i<lda;i++)
		c.mat[i] += a.mat[i+ldaj];
	}
    }
    else{
	fprintf(stderr,"sum_rows: dimension error\n");
	exit(1);
    }
}

void sum_cols( matrix a, matrix c, int omp_threads){
    int i,j;
    
    if(a.dim[1] == c.dim[1] && c.dim[0] == 1){

	const int lda = a.dim[0];
	const int tda = a.dim[1];

	for(j=0;j<tda;j++)
	    c.mat[j] = 0;
	
	for(j=0;j<tda;j++){
	    const int ldaj = lda*j;
	    for(i=0;i<lda;i++)
		c.mat[j] += a.mat[i+ldaj];
	}
    }
    else{
	fprintf(stderr,"sum_cols: dimension error\n");
	exit(1);
    }
}



