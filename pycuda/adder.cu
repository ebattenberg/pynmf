
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "adder.h"


#define BLOCK_SIZE 128

matrix mat_from_pyarray(PyObject* arr)
{
    matrix A;
    A.mat = (float*)PyArray_DATA(arr);
    A.mat_d = NULL;
    A.dim[0] = (int)PyArray_DIM(arr,0);
    A.dim[1] = (int)PyArray_DIM(arr,1);

    return A;
}


static PyObject* Add(PyObject* self, PyObject* args)
{
    PyObject *arg1=NULL, *arg2=NULL;
    PyObject *npy_a=NULL, *npy_b=NULL, *npy_c=NULL;
    if(!PyArg_ParseTuple(args, "OO", &arg1, &arg2))
	return NULL;
    if (arg1 == NULL) printf("arg1 NULL\n");

    // convert to contiguous arrays
    npy_a = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_FARRAY);
    if (npy_a == NULL) goto fail;
    npy_b = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_IN_FARRAY);
    if (npy_b == NULL) goto fail;
    npy_c = PyArray_EMPTY(PyArray_NDIM(npy_b), PyArray_DIMS(npy_b), NPY_FLOAT, 1);
    if (npy_c == NULL) goto fail;

    matrix a, b, c;


    a = mat_from_pyarray(npy_a);
    b = mat_from_pyarray(npy_b);
    c = mat_from_pyarray(npy_c);
    copy_matrix_to_device(&a);
    copy_matrix_to_device(&b);
    copy_matrix_to_device(&c);

    element_add(a,b,c);
    copy_matrix_from_device(&c);

    Py_DECREF(npy_a);
    Py_DECREF(npy_b);

    return npy_c;


fail:
    fprintf(stderr,"failed to allocate numpy arrays\n");
    return NULL;

}


PyMethodDef adder_methods[] = 
{
	{"add", Add, METH_VARARGS, "Add numbers"},
	{NULL,NULL,0,NULL} 
};

PyMODINIT_FUNC initadder(void)
{
    Py_InitModule("adder", adder_methods);
    import_array();
}


__global__ void vecAdd(float* a, float* b, float* c, const int N);

void element_add(matrix a, matrix b, matrix c)
{
    if (a.dim[0] != b.dim[0] || a.dim[1] != b.dim[1] ||
	    a.dim[0] != c.dim[0] || a.dim[1] != c.dim[1])
    {
	fprintf(stderr,"element_add: dimension mismatch\n");
	exit(1);
    }

    const int N = a.dim[0]*a.dim[1];

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N/dimBlock.x) + (!(N%dimBlock.x)?0:1));
    vecAdd<<<dimGrid,dimBlock>>>(a.mat_d,b.mat_d,c.mat_d,N);


}

__global__ void vecAdd(float* a, float* b, float* c, const int N)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N)
	c[i] = a[i] + b[i];
}


void copy_matrix_to_device(matrix* A)
{

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

void copy_matrix_from_device(matrix* A)
{

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













