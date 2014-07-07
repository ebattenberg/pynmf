

#include <Python.h>
#include <numpy/arrayobject.h>
#include "../mkl/matrix.h"
#include "run_nmf.h"


matrix matrix_from_pyarray(PyObject* arr);


static PyObject* Nmf(PyObject* self, PyObject* args)
{
    PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL;
    PyObject *npy_a=NULL, *npy_b=NULL, *npy_c=NULL;

    Py_ssize_t num_args = PyTuple_GET_SIZE(args);
    
    int max_iter; //max number of nmf iterations
    int threads; //number of OpenMP threads to use
    int verbose; //[0,1] print iteration information

    switch (num_args)
    {
	case 3:
	    if(!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3))
		return NULL;
	    break;
	case 4:
	    if(!PyArg_ParseTuple(args, "OOOi", &arg1, &arg2, &arg3, &max_iter))
		return NULL;
	    break;
	case 5:
	    if(!PyArg_ParseTuple(args, "OOOii", &arg1, &arg2, &arg3, &max_iter, &verbose))
		return NULL;
	    break;
	case 6:
	    if(!PyArg_ParseTuple(args, "OOOiii", &arg1, &arg2, &arg3, &max_iter, &verbose, &threads))
		return NULL;
	    break;
	default:
	    fprintf(stderr, "nmf: incorrect number of args\n");
	    return NULL;
    }
    if (num_args < 6)
	threads = 0;
    if (num_args < 5)
	verbose = 0;
    if (num_args < 4)
	max_iter = 500;
   


    // convert to contiguous arrays
    npy_a = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_FARRAY);
    if (npy_a == NULL) goto fail;
    npy_b = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_INOUT_FARRAY);
    if (npy_b == NULL) goto fail;
    npy_c = PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_INOUT_FARRAY);
    if (npy_c == NULL) goto fail;

    matrix X, W, H;


    X = matrix_from_pyarray(npy_a);
    W = matrix_from_pyarray(npy_b);
    H = matrix_from_pyarray(npy_c);


    run_nmf(X,W,H,threads,max_iter,verbose);
    
    return Py_None;


fail:
    fprintf(stderr,"failed to allocate numpy arrays\n");
    return NULL;

}




matrix matrix_from_pyarray(PyObject* arr)
{
    matrix A;
    A.mat = (float*)PyArray_DATA(arr);
    A.dim[0] = (int)PyArray_DIM(arr,0);
    A.dim[1] = (int)PyArray_DIM(arr,1);

    return A;
}


PyMethodDef nmf_methods[] = {
	{"nmf", Nmf, METH_VARARGS, "Perform non-negative matrix factorization"},
	{NULL,NULL,0,NULL} 
};

PyMODINIT_FUNC initnmf(void)
{
    Py_InitModule("nmf", nmf_methods);
    import_array();
}


