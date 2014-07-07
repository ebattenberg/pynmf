

#CC = gcc-4.3
CC = icc
CFLAGS = -O3 

MKLPATH = /opt/intel/mkl/10.1.1.019/lib/em64t
MKLINCLUDE = /opt/intel/mkl/10.1.1.019/include

ifeq (${CC},icc)
    CFLAGS += -openmp
    #LIBS = -lmkl
    LIBS = -lmkl_em64t
else
    CFLAGS += -fopenmp
    LIBS = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_lapack -lm
    #LIBS = $(MKLPATH)/libmkl_intel_lp64.so $(MKLPATH)/libmkl_gnu_thread.so $(MKLPATH)/libmkl_core.so -liomp5 -lm
    #LIBS = -lmkl
    #LIBS = -lmkl_em64t -liomp5 -lm
endif

#LIBS = -lblas -lm
INCLUDES = matrix.h
OBJS = matrix.o
SRCS = 
TARGETS = bench
OMP_THREADS = 2
MKL_THREADS = 2
#DEFS = -D OMP_THREADS=${OMP_THREADS} -D MKL_THREADS=${MKL_THREADS}


#MKLINCLUDE = /usr/lib/gcc/x86_64-linux-gnu/4.3/include


all: ${TARGETS} 

bench: bench.o ${OBJS}
	${CC} ${CFLAGS} -o $@ bench.o ${OBJS} -L$(MKLPATH) -I$(MKLINCLUDE) ${LIBS} 

clean:
	rm -f *.o ${TARGETS} *.prof *.out test bench

.c.o:
	${CC} ${CFLAGS} ${DEFS} -c -o $@ $< -L$(MKLPATH) -I$(MKLINCLUDE) ${LIBS} 

sgemm_bench: sgemm_bench.o ${OBJS}
	${CC} ${CFLAGS} -static -o $@ sgemm_bench.o ${OBJS} -L$(MKLPATH) -I$(MKLINCLUDE) ${LIBS} 


# The variable $@ has the value of the target. In this case $@ = psort
# prog1: prog1.o ${OBJS}
#
