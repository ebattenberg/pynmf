

#CC = gcc-4.3
CC = icc
#KERNELCC = gcc-4.3
KERNELCC = icc
#PYFLAGS = -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
PYFLAGS = -fPIC -O3 
PYINCLUDE = /usr/include/python2.6
LINKFLAGS = -fPIC -shared -openmp -O3 
KERNELFLAGS = -O3 -fPIC -openmp 

#MKLPATH = /opt/intel/mkl/10.0.2.018/lib/em64t
MKLPATH = $$MKLROOT/lib/em64t
#MKLINCLUDE = /opt/intel/mkl/10.1.0.015/include

ifeq (${KERNELCC},icc)
    KERNELFLAGS += -openmp
    KERNELLIBS = -lmkl -lmkl_def -lmkl_mc
else
    KERNELFLAGS += -fopenmp
    KERNELLIBS = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lm
endif

PYLIBS = $(KERNELLIBS)


OBJS = matrix.o run_nmf.o
TARGET = nmf


#MKLINCLUDE = /usr/lib/gcc/x86_64-linux-gnu/4.3/include

OBJS += $(TARGET).o
OBJDIR = build/temp/
OBJS := $(addprefix $(OBJDIR),$(OBJS)) 

VPATH = ../mkl


all: $(TARGET).so 

$(TARGET).so: $(OBJS)
	$(CC) $(LINKFLAGS) $^ -o $@ -L$(MKLPATH) $(PYLIBS) 

$(OBJDIR)$(TARGET).o: $(TARGET).c build_dirs
	$(CC) $(PYFLAGS) $(PYLIBS) -I$(PYINCLUDE) -c $< -o $@

$(OBJDIR)matrix.o: matrix.c build_dirs
	$(KERNELCC) $(KERNELFLAGS) $(KERNELLIBS) -c $< -o $@


$(OBJDIR)run_nmf.o: run_nmf.c build_dirs
	$(KERNELCC) $(KERNELFLAGS) $(KERNELLIBS) -c $< -o $@

$(OBJDIR)nmf_temp.o: $(OBJDIR)run_nmf.o $(OBJDIR)matrix.o
	ld -r -o $@ $^ -L$(MKLPATH) -ftz -lmkl_em64t


build_dirs: build/temp build/lib

build: 
	mkdir -p build

build/temp: build
	mkdir -p build/temp

build/lib: build
	mkdir -p build/lib




clean:
	rm -fr build/temp 

remove: 
	rm -fr build

