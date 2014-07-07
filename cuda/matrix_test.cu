#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "matrix.h"

#define EPS 2.2204E-16


__global__ void vecEps(float* a,const int N);
__global__ void vecDiv(float* a,float* b,float* c,const int N);
__global__ void vecMult(float* a,float* b,float* c,const int N);
__global__ void colDiv(float* a, float* b, float* c, int M, int N);
__global__ void colMul(float* a, float* b, float* c, int M, int N);
__global__ void rowDiv(float* a, float* b, float* c, int M, int N);
template <unsigned int blockSize>
__global__ void reduce2D(float *g_idata, float *g_odata, int N);
template <unsigned int blockSize>
__global__ void reduce2DStrided(float *g_idata, float *g_odata, int N, int stride);
template <unsigned int blockSize>
__global__ void reduce1DDiff(float *g_idata1, float *g_idata2, float *g_odata, int N);
template <unsigned int blockSize>
__global__ void reduce1DDiv(float *g_idata1, float *g_idata2, float *g_odata, int N);
template <unsigned int blockSize>
__global__ void reduce1D(float *g_idata, float *g_odata, int N);
template <unsigned int blockSize>
__global__ void reduce1D_rolled(float *g_idata, float *g_odata, int N);
template <unsigned int blockSize>
__global__ void reduce1D_rolled2(float *g_idata, float *g_odata, int N);
template <unsigned int blockSize>
__global__ void reduce1D_strided(float *g_idata, float *g_odata, int N);
template <unsigned int blockSize>
__global__ void reduce1D_interleaved(float *g_idata, float *g_odata, int N);


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
    cudaMallocHost((void**)&(A->mat),sizeof(float)*N);
    //A->mat = (float*)malloc(sizeof(float)*A->dim[0]*A->dim[1]);
    count = fread(A->mat,sizeof(float),N,fp);
    if(count < N)
	fprintf(stderr,"read_matrix: fread error\n");
    fclose(fp);

    A->mat_d = NULL;
    copy_matrix_to_device(A);

    printf("read %s [%ix%i]\n",file,A->dim[0],A->dim[1]); 
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
    //set A->mat_d to NULL
    
    A->dim[0] = rows;
    A->dim[1] = cols;
    const int N = A->dim[0]*A->dim[1];

    A->mat = (float*)malloc(sizeof(float)*N);
    for(int i = 0; i<N; i++)
	A->mat[i] = value;

    if(A->mat_d != NULL)
	cudaFree(A->mat_d);

    A->mat_d = NULL;
}

void create_matrix_on_device(matrix* A, int rows, int cols, float value){
    //create matrix on device  with all elements equal to 'value'
    //matrix dimensions are in dim[] {rows,cols}

    A->dim[0] = rows;
    A->dim[1] = cols;
    A->mat = NULL;

    const int N = A->dim[0]*A->dim[1];

    cudaError_t err;
    err = cudaMalloc((void**) &(A->mat_d), sizeof(float)*N);
    if (err != cudaSuccess){
	fprintf(stderr,"create_matrix_on_device: cudaMalloc: ErrorMemoryAllocation\n");
	exit(1);
    }

    float *temp = (float*)malloc(sizeof(float)*N);
    for(int i = 0; i<N; i++)
	temp[i] = value;
    cudaMemcpy(A->mat_d,temp,sizeof(float)*N,cudaMemcpyHostToDevice);

    free(temp);


}

void create_matrix_on_both(matrix* A, int rows, int cols, float value){
    //create matrix on device  with all elements equal to 'value'
    //matrix dimensions are in dim[] {rows,cols}

    A->dim[0] = rows;
    A->dim[1] = cols;
    const int N = A->dim[0]*A->dim[1];
    cudaError_t err;


    err = cudaMalloc((void**) &(A->mat_d), sizeof(float)*N);
    if (err != cudaSuccess){
	fprintf(stderr,"create_matrix_on_both: cudaMalloc: ErrorMemoryAllocation\n");
	exit(1);
    }

    A->mat = (float*)malloc(sizeof(float)*N);
    for(int i = 0; i<N; i++)
	A->mat[i] = value;
    cudaMemcpy(A->mat_d,A->mat,sizeof(float)*N,cudaMemcpyHostToDevice);

}

void destroy_matrix(matrix* A){
    if(A->mat != NULL)
	cudaFreeHost(A->mat);
    A->mat = NULL;
    if(A->mat_d != NULL)
	cudaFree(A->mat_d);
    A->mat_d = NULL;

    A->dim[0] = 0;
    A->dim[1] = 0;
}

void copy_matrix_to_device(matrix* A){

    const int N = A->dim[0]*A->dim[1];
    cudaError_t err;

    if (A->mat == NULL){
	fprintf(stderr,"copy_matrix_to_device: matrix not allocated on host\n");
	exit(1);
    }
    if (A->mat_d == NULL){
	err = cudaMalloc((void**) &(A->mat_d), sizeof(float)*N);
	if(err != cudaSuccess){
	    fprintf(stderr,"copy_matrix_to_device: cudaMalloc: FAIL\n");
	    exit(1);
	}
    }

    err = cudaMemcpy(A->mat_d,A->mat,sizeof(float)*N, cudaMemcpyHostToDevice);
    switch (err){
	case cudaErrorInvalidValue:
	fprintf(stderr,"copy_matrix_to_device: cudaMemcpy: InvalidValue\n");
	exit(1);
	break;
	case cudaErrorInvalidDevicePointer:
	fprintf(stderr,"copy_matrix_to_device: cudaMemcpy: InvalidDevicePointer\n");
	exit(1);
	break;
	case cudaErrorInvalidMemcpyDirection:
	fprintf(stderr,"copy_matrix_to_device: cudaMemcpy: InvalidMemcpyDirection\n");
	exit(1);
	break;
    }
}

void copy_matrix_on_device(matrix A, matrix B){

    if(A.dim[0]!=B.dim[0] || A.dim[1]!=B.dim[1]){
	fprintf(stderr,"copy_matrix_on_device: dimension error\n");
	exit(1);
    }
    const int N = A.dim[0]*A.dim[1];
    cudaError_t err;

    if (A.mat_d == NULL){
	fprintf(stderr,"copy_matrix_on_device: source matrix not allocated on device\n");
	exit(1);
    }
    if (B.mat_d == NULL){
	fprintf(stderr,"copy_matrix_on_device: dest. matrix not allocated on device\n");
	exit(1);
    }

    err = cudaMemcpy(B.mat_d,A.mat_d,sizeof(float)*N, cudaMemcpyDeviceToDevice);
    switch (err){
	case cudaErrorInvalidValue:
	fprintf(stderr,"copy_matrix_on_device: cudaMemcpy: InvalidValue\n");
	exit(1);
	break;
	case cudaErrorInvalidDevicePointer:
	fprintf(stderr,"copy_matrix_on_device: cudaMemcpy: InvalidDevicePointer\n");
	exit(1);
	break;
	case cudaErrorInvalidMemcpyDirection:
	fprintf(stderr,"copy_matrix_on_device: cudaMemcpy: InvalidMemcpyDirection\n");
	exit(1);
	break;
    }
}

void copy_matrix_from_device(matrix* A){

    const int N = A->dim[0]*A->dim[1];

    if (A->mat_d == NULL){
	fprintf(stderr,"copy_matrix_from_device: matrix not allocated on device\n");
	exit(1);
    }
    if (A->mat == NULL)
	cudaMallocHost((void**)&(A->mat),sizeof(float)*N);
	//A->mat = (float*)malloc(sizeof(float)*N);

    cudaError_t err;
    err = cudaMemcpy(A->mat,A->mat_d,sizeof(float)*N, cudaMemcpyDeviceToHost);
    switch (err){
	case cudaErrorInvalidValue:
	fprintf(stderr,"copy_matrix_to_device: cudaMemcpy: InvalidValue\n");
	exit(1);
	break;
	case cudaErrorInvalidDevicePointer:
	fprintf(stderr,"copy_matrix_to_device: cudaMemcpy: InvalidDevicePointer\n");
	exit(1);
	break;
	case cudaErrorInvalidMemcpyDirection:
	fprintf(stderr,"copy_matrix_to_device: cudaMemcpy: InvalidMemcpyDirection\n");
	exit(1);
	break;
    }
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

void matrix_multiply_d( matrix a, matrix b, matrix c ){
    
    cublasSgemm('N',
	    'N', c.dim[0], c.dim[1],
	    a.dim[1], 1, a.mat_d,
	    a.dim[0], b.mat_d, b.dim[0],
	    0, c.mat_d, c.dim[0]);
    if(cublasGetError() != CUBLAS_STATUS_SUCCESS){
	fprintf(stderr,"matrix_multiply_d: NOT SUCCESS\n"); 
	exit(1);
    }
}

void matrix_multiply_AtB_d( matrix a, matrix b, matrix c ){
    
    cublasSgemm('T',
	    'N', c.dim[0], c.dim[1],
	    b.dim[0], 1, a.mat_d,
	    a.dim[0], b.mat_d, b.dim[0],
	    0, c.mat_d, c.dim[0]);
    if(cublasGetError() != CUBLAS_STATUS_SUCCESS){
	fprintf(stderr,"matrix_multiply_AtB_d: NOT SUCCESS\n"); 
	exit(1);
    }
}

void matrix_multiply_ABt_d( matrix a, matrix b, matrix c ){
    
    cublasSgemm('N',
	    'T', c.dim[0], c.dim[1],
	    a.dim[1], 1, a.mat_d,
	    a.dim[0], b.mat_d, b.dim[0],
	    0, c.mat_d, c.dim[0]);
    cublasStatus err = cublasGetError();
    if(err != CUBLAS_STATUS_SUCCESS){
	fprintf(stderr,"matrix_multiply_ABt_d: NOT SUCCESS [%i]\n",err); 
	switch(err){
	    case CUBLAS_STATUS_NOT_INITIALIZED:
		fprintf(stderr,"CUBLAS_STATUS_NOT_INITIALIZED\n");
		break;
	    case CUBLAS_STATUS_ALLOC_FAILED:
		fprintf(stderr,"CUBLAS_STATUS_ALLOC_FAILED\n");
		break;
	    case CUBLAS_STATUS_INVALID_VALUE:
		fprintf(stderr,"CUBLAS_STATUS_INVALID_VALUE\n");
		break;
	    case CUBLAS_STATUS_MAPPING_ERROR:
		fprintf(stderr,"CUBLAS_STATUS_MAPPING_ERROR\n");
		break;
	    case CUBLAS_STATUS_EXECUTION_FAILED:
		fprintf(stderr,"CUBLAS_STATUS_EXECUTION_FAILED\n");
		break;
	}
	exit(1);
    }
}

float matrix_difference_norm_d(action_t action, matrix a, matrix b, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    static int r1size = 0;
    static float *r1 = NULL;
    static float *result_d = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	if(result_d!=NULL){
	    cudaFree(result_d);
	    result_d = NULL;
	}
	r1size = 0;
	return 0;
    }
    
    if(a.dim[0] != b.dim[0] || a.dim[1] != b.dim[1]){
	fprintf(stderr,"matrix_difference_norm_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0]*a.dim[1];	//size of each reduction
    
    dim3 dimBlock(block1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1));

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),2);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if(result_d == NULL)
	cudaMalloc((void**) &result_d, sizeof(float)*2);
    if (r1size < dimGrid.x*2){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*2;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"matrix_difference_norm_d: dimGrid.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce1DDiff<512><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 256:
		reduce1DDiff<256><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 128:
		reduce1DDiff<128><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 64:
		reduce1DDiff<64><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 32:
		reduce1DDiff<32><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 16:
		reduce1DDiff<16><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 8:
		reduce1DDiff<8><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	}
    }
    else{ //if we need two levels of reduction
	if (dimGrid2.x > 1){
	    fprintf(stderr,"matrix_difference_norm_d: dimGrid2.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce1DDiff<512><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 256:
		reduce1DDiff<256><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 128:
		reduce1DDiff<128><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 64:
		reduce1DDiff<64><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 32:
		reduce1DDiff<32><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 16:
		reduce1DDiff<16><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 8:
		reduce1DDiff<8><<< dimGrid, dimBlock, 2*dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	}
	switch (block2)
	{
	    case 512:
		reduce2D<512><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 256:
		reduce2D<256><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 128:
		reduce2D<128><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 64:
		reduce2D<64><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 32:
		reduce2D<32><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 16:
		reduce2D<16><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 8:
		reduce2D<8><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	}
    }

    float result[2];
    cudaMemcpy(result,result_d,2*sizeof(float),cudaMemcpyDeviceToHost);
    return result[0]/result[1];



}

float matrix_div_d(action_t action, matrix a, matrix b, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    
    static int r1size = 0;
    static float *r1 = NULL;
    static float *result_d = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	if(result_d!=NULL){
	    cudaFree(result_d);
	    result_d = NULL;
	}
	r1size = 0;
	return 0;
    }

    if(a.dim[0] != b.dim[0] || a.dim[1] != b.dim[1]){
	fprintf(stderr,"matrix_div_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0]*a.dim[1];	//size of each reduction
    
    dim3 dimBlock(block1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1));

    dim3 dimBlock2(block2);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1));

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if(result_d == NULL)
	cudaMalloc((void**) &result_d, sizeof(float)*1);
    if (r1size < dimGrid.x){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"matrix_difference_norm_d: dimGrid.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce1DDiv<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 256:
		reduce1DDiv<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 128:
		reduce1DDiv<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 64:
		reduce1DDiv<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 32:
		reduce1DDiv<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 16:
		reduce1DDiv<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	    case 8:
		reduce1DDiv<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,result_d,N); break;
	}
    }
    else{ //if we need two levels of reduction
	if (dimGrid2.x > 1){
	    fprintf(stderr,"matrix_difference_norm_d: dimGrid2.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce1DDiv<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 256:
		reduce1DDiv<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 128:
		reduce1DDiv<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 64:
		reduce1DDiv<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 32:
		reduce1DDiv<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 16:
		reduce1DDiv<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	    case 8:
		reduce1DDiv<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,b.mat_d,r1,N); break;
	}
	switch (block2)
	{
	    case 512:
		reduce2D<512><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 256:
		reduce2D<256><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 128:
		reduce2D<128><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 64:
		reduce2D<64><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 32:
		reduce2D<32><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 16:
		reduce2D<16><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	    case 8:
		reduce2D<8><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,result_d,dimGrid.x); break;
	}
    }

    float result;
    cudaMemcpy(&result,result_d,1*sizeof(float),cudaMemcpyDeviceToHost);
    return result;



}

void element_divide_d( matrix a, matrix b, matrix c, int block_size){
    // c = a./b

    if(a.dim[0] != b.dim[0] || a.dim[0] != c.dim[0] ||
	    a.dim[1] != b.dim[1] || a.dim[1] != c.dim[1])
    {
	fprintf(stderr,"element_divide_d: dimensions do not agree\n");
	exit(1);
    }

    const int N = a.dim[0]*a.dim[1];
    dim3 dimBlock(block_size);
    dim3 dimGrid((N/dimBlock.x) + (!(N%dimBlock.x)?0:1));
    vecDiv<<<dimGrid,dimBlock>>>(a.mat_d,b.mat_d,c.mat_d,N);
}
    
__global__ void vecDiv(float* a,float* b,float* c,const int N){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N)
	c[i] = __fdividef(a[i],b[i]);
	//c[i] = a[i]/b[i];
}

void element_multiply_d( matrix a, matrix b, matrix c){
    // c = a./b

    if(a.dim[0] == b.dim[0] && a.dim[0] == c.dim[0] &&
	    a.dim[1] == b.dim[1] && a.dim[1] == c.dim[1])
    {
	const int N = a.dim[0]*a.dim[1];
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((N/dimBlock.x) + (!(N%dimBlock.x)?0:1));
	vecMult<<<dimGrid,dimBlock>>>(a.mat_d,b.mat_d,c.mat_d,N);
    }
    else{
	fprintf(stderr,"element_multiply_d: dimensions do not agree\n");
	exit(1);
    }
}
    
__global__ void vecMult(float* a,float* b,float* c,const int N){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N)
	c[i] = a[i]*b[i];
}

__global__ void vecEps(float* a,const int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(a[i] < EPS && i < N)
	a[i] = EPS;
}

void matrix_eps_d(matrix a, int block_size){

    const int N = a.dim[0]*a.dim[1];

    dim3 dimBlock(block_size);
    dim3 dimGrid((N/dimBlock.x) + (!(N%dimBlock.x)?0:1));
    vecEps<<<dimGrid, dimBlock>>>(a.mat_d,N);
}

void row_divide_d( matrix a, matrix b, matrix c){
    //element divide every row of 'a' by row vector 'b'

    if(a.dim[1] != b.dim[1] || a.dim[0] != c.dim[0] ||
	    a.dim[1] != c.dim[1] || b.dim[0] != 1){
	fprintf(stderr,"row_divide_d: dimension error\n");
	exit(1);
    }
    int M = a.dim[0]; //number of rows
    int N = a.dim[1]; //number of cols

    dim3 dimBlock(M);
    dim3 dimGrid(N);
    rowDiv<<<dimGrid,dimBlock>>>(a.mat_d,b.mat_d,c.mat_d,M,N);
}

__global__ void rowDiv(float* a, float* b, float* c, int M, int N){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = a[i]/b[blockIdx.x];
}

void col_divide_d( matrix a, matrix b, matrix c){
    //element divide every column of 'a' by column vector 'b'

    if(a.dim[0] != b.dim[0] || a.dim[0] != c.dim[0] ||
	    a.dim[1] != c.dim[1] || b.dim[1] != 1){
	fprintf(stderr,"col_divide: dimension error\n");
	exit(1);
    }
    int M = a.dim[0]; //number of rows
    int N = a.dim[1]; //number of cols
    int block = 32;

    dim3 dimBlock(block,1);
    dim3 dimGrid((M/block) + (!(M%block)?0:1),N);
    colDiv<<<dimGrid,dimBlock>>>(a.mat_d,b.mat_d,c.mat_d,M,N);

}

__global__ void colDiv(float* a, float* b, float* c, int M, int N){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<M){
	int ind = i + blockIdx.y*M;
	c[ind] = a[ind]/b[i];
    }
}

__global__ void colMul(float* a, float* b, float* c, int M, int N){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<M){
	int ind = i + blockIdx.y*M;
	c[ind] = a[ind]*b[i];
    }
}

void sum_cols_d(action_t action, matrix a, matrix c, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	r1size = 0;
	return;
    }
    
    if(a.dim[1] != c.dim[1] || c.dim[0] != 1){
	fprintf(stderr,"sum_cols_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0];	//size of each reduction
    const int M = a.dim[1];	//number of reductions
    
    dim3 dimBlock(block1,1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1),M);

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),M);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if (r1size < dimGrid.x*dimGrid.y){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*dimGrid.y;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce2D<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N); break;
	    case 256:
		reduce2D<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N); break;
	    case 128:
		reduce2D<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N); break;
	    case 64:
		reduce2D<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N); break;
	    case 32:
		reduce2D<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N); break;
	    case 16:
		reduce2D<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N); break;
	    case 8:
		reduce2D<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N); break;
	}
    }
    else{ //if we need two levels of reduction
	if (dimGrid2.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid2.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce2D<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 256:
		reduce2D<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 128:
		reduce2D<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 64:
		reduce2D<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 32:
		reduce2D<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 16:
		reduce2D<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 8:
		reduce2D<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	}
	switch (block2)
	{
	    case 512:
		reduce2D<512><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 256:
		reduce2D<256><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 128:
		reduce2D<128><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 64:
		reduce2D<64><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 32:
		reduce2D<32><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 16:
		reduce2D<16><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 8:
		reduce2D<8><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	}
    }


}

void sum_cols_d_test(action_t action, matrix a, matrix c, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	r1size = 0;
	return;
    }
    
    if(a.dim[1] != c.dim[1] || c.dim[0] != 1){
	fprintf(stderr,"sum_cols_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0];	//size of each reduction
    const int M = a.dim[1];	//number of reductions
    
    dim3 dimBlock(block1,1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1),1);

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),1);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if (r1size < dimGrid.x*dimGrid.y){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*dimGrid.y;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid.x > 1\n");
	    exit(1);
	}
	for(int i=0;i<M;i++)
	{
	    switch (block1)
	    {
		case 512:
		    reduce1D<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 256:
		    reduce1D<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 128:
		    reduce1D<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 64:
		    reduce1D<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 32:
		    reduce1D<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 16:
		    reduce1D<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 8:
		    reduce1D<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
	    }
	}
    }
    else{ //if we need two levels of reduction
	if (dimGrid2.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid2.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce2D<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 256:
		reduce2D<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 128:
		reduce2D<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 64:
		reduce2D<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 32:
		reduce2D<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 16:
		reduce2D<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	    case 8:
		reduce2D<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N); break;
	}
	switch (block2)
	{
	    case 512:
		reduce2D<512><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 256:
		reduce2D<256><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 128:
		reduce2D<128><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 64:
		reduce2D<64><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 32:
		reduce2D<32><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 16:
		reduce2D<16><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	    case 8:
		reduce2D<8><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x); break;
	}
    }


}

void sum_cols_d_test2(action_t action, matrix a, matrix c, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	r1size = 0;
	return;
    }
    
    if(a.dim[1] != c.dim[1] || c.dim[0] != 1){
	fprintf(stderr,"sum_cols_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0];	//size of each reduction
    const int M = a.dim[1];	//number of reductions
    
    dim3 dimBlock(block1,1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1),1);

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),1);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if (r1size < dimGrid.x*dimGrid.y){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*dimGrid.y;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid.x > 1\n");
	    exit(1);
	}
	for(int i=0;i<M;i++)
	{
	    switch (block1)
	    {
		case 512:
		    reduce1D_rolled<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 256:
		    reduce1D_rolled<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 128:
		    reduce1D_rolled<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 64:
		    reduce1D_rolled<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 32:
		    reduce1D_rolled<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 16:
		    reduce1D_rolled<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 8:
		    reduce1D_rolled<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
	    }
	}
    }


}

void sum_cols_d_test3(action_t action, matrix a, matrix c, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	r1size = 0;
	return;
    }
    
    if(a.dim[1] != c.dim[1] || c.dim[0] != 1){
	fprintf(stderr,"sum_cols_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0];	//size of each reduction
    const int M = a.dim[1];	//number of reductions
    
    dim3 dimBlock(block1,1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1),1);

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),1);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if (r1size < dimGrid.x*dimGrid.y){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*dimGrid.y;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid.x > 1\n");
	    exit(1);
	}
	for(int i=0;i<M;i++)
	{
	    switch (block1)
	    {
		case 512:
		    reduce1D_rolled2<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 256:
		    reduce1D_rolled2<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 128:
		    reduce1D_rolled2<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 64:
		    reduce1D_rolled2<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 32:
		    reduce1D_rolled2<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 16:
		    reduce1D_rolled2<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 8:
		    reduce1D_rolled2<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
	    }
	}
    }


}

void sum_cols_d_test4(action_t action, matrix a, matrix c, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	r1size = 0;
	return;
    }
    
    if(a.dim[1] != c.dim[1] || c.dim[0] != 1){
	fprintf(stderr,"sum_cols_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0];	//size of each reduction
    const int M = a.dim[1];	//number of reductions
    
    dim3 dimBlock(block1,1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1),1);

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),1);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if (r1size < dimGrid.x*dimGrid.y){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*dimGrid.y;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid.x > 1\n");
	    exit(1);
	}
	for(int i=0;i<M;i++)
	{
	    switch (block1)
	    {
		case 512:
		    reduce1D_strided<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 256:
		    reduce1D_strided<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 128:
		    reduce1D_strided<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 64:
		    reduce1D_strided<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 32:
		    reduce1D_strided<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 16:
		    reduce1D_strided<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 8:
		    reduce1D_strided<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
	    }
	}
    }


}

void sum_cols_d_test5(action_t action, matrix a, matrix c, int block1, int lapt1,int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	r1size = 0;
	return;
    }
    
    if(a.dim[1] != c.dim[1] || c.dim[0] != 1){
	fprintf(stderr,"sum_cols_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[0];	//size of each reduction
    const int M = a.dim[1];	//number of reductions
    
    dim3 dimBlock(block1,1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1),1);

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),1);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if (r1size < dimGrid.x*dimGrid.y){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*dimGrid.y;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"sum_cols_d: dimGrid.x > 1\n");
	    exit(1);
	}
	for(int i=0;i<M;i++)
	{
	    switch (block1)
	    {
		case 512:
		    reduce1D_interleaved<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 256:
		    reduce1D_interleaved<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 128:
		    reduce1D_interleaved<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 64:
		    reduce1D_interleaved<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 32:
		    reduce1D_interleaved<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 16:
		    reduce1D_interleaved<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
		case 8:
		    reduce1D_interleaved<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(&a.mat_d[N*i],&c.mat_d[i],N); break;
	    }
	}
    }


}

void sum_rows_d(action_t action, matrix a, matrix c, int block1, int lapt1, int block2, int lapt2){
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt1 - load/adds per thread for first red. lev.
    //lapt2 - "" for 2nd ""
    
    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup){
	if(r1!=NULL){
	    cudaFree(r1);
	    r1 = NULL;
	}
	r1size = 0;
	return;
    }
    if(a.dim[0] != c.dim[0] || c.dim[1] != 1){
	fprintf(stderr,"sum_rows_d: dimension error\n");
	exit(1);
    }

    const int N = a.dim[1];	//size of each reduction
    const int M = a.dim[0];	//number of reductions
    
    dim3 dimBlock(block1,1);
    dim3 dimGrid((N/(block1*lapt1)) + (!(N%(block1*lapt1))?0:1),M);

    dim3 dimBlock2(block2,1);
    dim3 dimGrid2((dimGrid.x/(block2*lapt2)) + (!(dimGrid.x%(block2*lapt2))?0:1),M);

    //printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    //printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    //allocate memory for first level reduction
    if (r1size < dimGrid.x*dimGrid.y){
	if(r1 != NULL)
	    cudaFree(r1);
	r1size = dimGrid.x*dimGrid.y;
	cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if(block2 <= 1){ //if we only need one level of reduction
	if (dimGrid.x > 1){
	    fprintf(stderr,"sum_rows_d: dimGrid.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce2DStrided<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N,M); break;
	    case 256:
		reduce2DStrided<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N,M); break;
	    case 128:
		reduce2DStrided<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N,M); break;
	    case 64:
		reduce2DStrided<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N,M); break;
	    case 32:
		reduce2DStrided<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N,M); break;
	    case 16:
		reduce2DStrided<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N,M); break;
	    case 8:
		reduce2DStrided<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,c.mat_d,N,M); break;
	}
    }
    else{ //if we need two levels of reduction
	if (dimGrid2.x > 1){
	    fprintf(stderr,"sum_rows_d: dimGrid2.x > 1\n");
	    exit(1);
	}
	switch (block1)
	{
	    case 512:
		reduce2DStrided<512><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N,M); break;
	    case 256:
		reduce2DStrided<256><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N,M); break;
	    case 128:
		reduce2DStrided<128><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N,M); break;
	    case 64:
		reduce2DStrided<64><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N,M); break;
	    case 32:
		reduce2DStrided<32><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N,M); break;
	    case 16:
		reduce2DStrided<16><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N,M); break;
	    case 8:
		reduce2DStrided<8><<< dimGrid, dimBlock, dimBlock.x*sizeof(float) >>>(a.mat_d,r1,N,M); break;
	}
	switch (block2)
	{
	    case 512:
		reduce2DStrided<512><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x,M); break;
	    case 256:
		reduce2DStrided<256><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x,M); break;
	    case 128:
		reduce2DStrided<128><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x,M); break;
	    case 64:
		reduce2DStrided<64><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x,M); break;
	    case 32:
		reduce2DStrided<32><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x,M); break;
	    case 16:
		reduce2DStrided<16><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x,M); break;
	    case 8:
		reduce2DStrided<8><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,c.mat_d,dimGrid.x,M); break;
	}
    }


}


template <unsigned int blockSize>
__global__ void reduce1DDiff(float *g_idata1, float *g_idata2, float *g_odata, int N){
    extern __shared__ float sdata[];
    float* diff = (float*)sdata;
    float* sum = (float*)&sdata[blockSize];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize + threadIdx.x;
    const int gridSize = blockSize*gridDim.x;
    sum[tid] = 0;
    diff[tid] = 0;
    while (i < N) { 
	diff[tid] += fabs(g_idata1[i] - g_idata2[i]);
	sum[tid] += fabs(g_idata1[i]);
       	i += gridSize; 
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { diff[tid] += diff[tid + 256]; sum[tid] += sum[tid + 256]; }
       	__syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { diff[tid] += diff[tid + 128]; sum[tid] += sum[tid + 128]; } 
	__syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { diff[tid] += diff[tid + 64]; sum[tid] += sum[tid + 64]; } 
	__syncthreads(); }
    if (tid < 32) {
	if (blockSize >= 64){ diff[tid] += diff[tid + 32]; sum[tid] += sum[tid + 32]; }
	if (blockSize >= 32){ diff[tid] += diff[tid + 16]; sum[tid] += sum[tid + 16]; }
	if (blockSize >= 16){ diff[tid] += diff[tid + 8]; sum[tid] += sum[tid + 8]; }
	if (blockSize >= 8){ diff[tid] += diff[tid + 4]; sum[tid] += sum[tid + 4]; }
	if (blockSize >= 4){ diff[tid] += diff[tid + 2]; sum[tid] += sum[tid + 2]; }
	if (blockSize >= 2){ diff[tid] += diff[tid + 1]; sum[tid] += sum[tid + 1]; }
    }

    // write result for this block to global mem
    if (tid == 0){ 
	g_odata[blockIdx.x + gridDim.x] = sum[0];
	g_odata[blockIdx.x] = diff[0];
    }
}

template <unsigned int blockSize>
__global__ void reduce1DDiv(float *g_idata1, float *g_idata2, float *g_odata, int N){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize + threadIdx.x;
    const int gridSize = blockSize*gridDim.x;
    float x;
    float y;
    sdata[tid] = 0;
    while (i < N) { 
	x = g_idata1[i];
	y = g_idata2[i];
	//sdata[tid] += (x*__logf(x/y)-x+y); 
	sdata[tid] += (x*(__logf(x)-__logf(y))-x+y); 
       	i += gridSize; 
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
       	__syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
	__syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
	__syncthreads(); }
    if (tid < 32) {
	if (blockSize >= 64){ sdata[tid] += sdata[tid + 32]; }
	if (blockSize >= 32){ sdata[tid] += sdata[tid + 16]; }
	if (blockSize >= 16){ sdata[tid] += sdata[tid + 8]; }
	if (blockSize >= 8){ sdata[tid] += sdata[tid + 4]; }
	if (blockSize >= 4){ sdata[tid] += sdata[tid + 2]; }
	if (blockSize >= 2){ sdata[tid] += sdata[tid + 1]; }
    }

    // write result for this block to global mem
    if (tid == 0){ 
	g_odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__global__ void reduce2D(float *g_idata, float *g_odata, int N){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int offset = blockIdx.y*N;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i+offset] + g_idata[i+offset+blockSize]; i += gridSize; }
    if(i<N)
	sdata[tid] += g_idata[i+offset];
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x + blockIdx.y*gridDim.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce1D(float *g_idata, float *g_odata, int N){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    if(i<N)
	sdata[tid] += g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce1D_rolled(float *g_idata, float *g_odata, int N){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    if(i<N)
	sdata[tid] += g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
	if (tid < s)
	    sdata[tid] += sdata[tid + s];
	__syncthreads();
    }
    if (tid < 32)
    {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce1D_rolled2(float *g_idata, float *g_odata, int N){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    if(i<N)
	sdata[tid] += g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
	if (tid < s)
	    sdata[tid] += sdata[tid + s];
	__syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce1D_strided(float *g_idata, float *g_odata, int N){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    if(i<N)
	sdata[tid] += g_idata[i];
    __syncthreads();
    // do reduction in shared mem

    for (unsigned int s=1; s < blockDim.x; s *= 2) {
	int index = 2 * s * tid;
	if (index < blockDim.x) {
	    sdata[index] += sdata[index + s];
	}
	__syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce1D_interleaved(float *g_idata, float *g_odata, int N){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    if(i<N)
	sdata[tid] += g_idata[i];
    __syncthreads();
    // do reduction in shared mem

    for (unsigned int s=1; s < blockDim.x; s *= 2) {
	if (tid % (2*s) == 0) {
	    sdata[tid] += sdata[tid + s];
	}
	__syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce2DStrided(float *g_idata, float *g_odata, int N, int stride){
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int offset = blockIdx.y;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;
    sdata[tid] = 0;
    while (i < n) { 
	sdata[tid] += g_idata[i*stride+offset] + g_idata[(i+blockSize)*stride+offset];
       	i += gridSize; 
    }
    if(i<N)
	sdata[tid] += g_idata[i*stride+offset];
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.y + blockIdx.x*gridDim.y] = sdata[0];
}



