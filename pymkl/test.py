#!/usr/bin/python

import numpy as np
import array
import nmf

#a = np.arange(9,dtype='float32').reshape((3,3),order='F');
#b = np.eye(3,dtype='float32')

def npy_from_file(filename):
    fp = open(filename,'rb')
    #fp = open('/home/ericb/school/parlab/working/parlab/nmf/trunk/H.bin','rb')
    dim = array.array('i') 		#init int array
    dim.fromfile(fp,2) 		#read in 2 int
    N = dim[0]*dim[1] 		#number of values to read
    binvals = array.array('f')	#init float array
    binvals.fromfile(fp,N)
    fp.close()
    A = np.array(binvals,'float32')
    #reshape using fortran (col-major) order
    A = A.reshape((dim[0],dim[1]),order='F') 
    return A

def npy_to_file(A,filename):
    fp = open(filename,'wb')
    dim = array.array('i',A.shape)
    dim.tofile(fp)
    binvals = array.array('f',A.flatten('F').tolist())
    binvals.tofile(fp)
    fp.close()
    return

path = '/home/ericb/projects/nmf/trunk/'
X = npy_from_file(path + 'X.bin')
W = npy_from_file(path + 'W.bin')
H = npy_from_file(path + 'H.bin')


#print X
#print W
#print H


nmf.nmf(X,W,H,100,1)


#print W
#print H
#print np.dot(W,H)








