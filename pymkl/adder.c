
#include <Python.h>
#include <numpy/arrayobject.h>


typedef struct {
    int dim[2];
    float *mat;
} matrix;

matrix mat_from_pyarray(PyObject* arr)
{
    matrix A;
    A.mat = (float*)PyArray_DATA(arr);
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
    npy_c = PyArray_FROM_OTF(npy_b, NPY_FLOAT, NPY_INOUT_FARRAY);
    if (npy_c == NULL) goto fail;

    matrix a, b, c;


    a = mat_from_pyarray(npy_a);
    b = mat_from_pyarray(npy_b);
    c = mat_from_pyarray(npy_c);




    int a_size = PyArray_SIZE(npy_a);

    int i;
    for(i=0;i<a_size;i++)
	c.mat[i] = a.mat[i] + b.mat[i];




    Py_DECREF(npy_a);
    Py_DECREF(npy_b);
    Py_INCREF(Py_None);
    return npy_c;


fail:
    fprintf(stderr,"failed to allocate numpy arrays\n");
    return NULL;

}


PyMethodDef adder_methods[] = {
	{"add", Add, METH_VARARGS, "Add numbers"},
	{NULL,NULL,0,NULL} 
};

PyMODINIT_FUNC initadder(void)
{
    Py_InitModule("adder", adder_methods);
    import_array();
}














