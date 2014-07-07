CFLAGS = -Wall -O3 -fopenmp


OMP_THREADS = 1 #number of threads for openmp to use
ATL_THREADED = 1 #1 to use threaded atlas, empty to use sequential atlas

ONE = one

CC = gcc-4.3

ifdef ATL_THREADED
    LIBS = -lptcblas -latlas -lm -lpthread
else
    LIBS = -lcblas -latlas -lm -lpthread
endif

CFLAGS = -Wall -O3 -fopenmp

INCLUDES = matrix.h
OBJS = matrix.o
SRCS = 
TARGETS = nmf
DEFS = -D OMP_THREADS=${OMP_THREADS}

ATLPATH = /usr/local/atlas/lib
ATLINCLUDE = /usr/local/atlas/include

#MKLINCLUDE = /usr/lib/gcc/x86_64-linux-gnu/4.3/include


all: ${TARGETS} 

nmf: nmf.o ${OBJS}
	${CC} ${CFLAGS} -o $@ nmf.o ${OBJS} ${LIBS} -L${ATLPATH} -I${ATLINCLUDE}

clean:
	rm -f *.o ${TARGETS} *.prof *.out

.c.o:
	${CC} ${CFLAGS} ${DEFS} -c -o $@ $< -L${ATLPATH} -I${ATLINCLUDE}

test: test.o ${OBJS}
	${CC} ${CFLAGS} -o $@ test.o ${OBJS} ${LIBS} -L${ATLPATH} -I${ATLINCLUDE}


# The variable $@ has the value of the target. In this case $@ = psort
# prog1: prog1.o ${OBJS}
#
