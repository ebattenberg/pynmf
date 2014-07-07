# vim:comments=
#
# I don't like the Cuda SDK makefiles. Here's my own! This makefile should
# allow you to build Cuda programs independent of the NVidia Cuda SDK. Its
# simplicity makes it easy for you to put objects, binaries, etc where you want
# them. There is even some boiler-plate Makefile magic that you can copy
# into other Makefiles.
#
# Mark Murphy 2008

# Fill in the name of the output binary here
target    := nmf.so

# List of sources, with .c, .cu, and .cc extensions
sources   := nmf.cu run_nmf.cu matrix.cu
#sources   := adder.c adder_d.cu

# Other things that need to be built, e.g. .cubin files
extradeps := 


# Flags common to all compilers. You can set these on the comamnd line, e.g:
# $ make opt="" dbg="-g" warn="-Wno-deptrcated-declarations -Wall -Werror"

opt  ?= -O3
dbg  ?= 
warn ?= -Wall -Werror


# This is where the cuda runtime libraries and includes reside on the
# berkeley millennium/rad clusters. It may be different on your machine.

#cudaroot  := /usr/mill/pkg/cuda
cudaroot  := /usr/local/cuda


#----- C compilation options ------
gcc        := /usr/bin/gcc
cflags     += $(opt) $(dbg) $(warn) -fPIC 
clib_paths :=

cinc_paths := -I $(cudaroot)/include -I/usr/include/python2.7
clibraries := -lcuda -lcudart -lblas


#----- C++ compilation options ------
gpp         := /usr/bin/g++
ccflags     += $(opt) $(dbg) $(warn)
cclib_paths :=
ccinc_paths := 
cclibraries :=

#----- CUDA compilation options -----

nvcc        := $(cudaroot)/bin/nvcc
cuflags     += $(opt) $(dbg) -Xcompiler -fPIC 
culib_paths := -L$(cudaroot)/lib
cuinc_paths := -I $(cudaroot)/include -I/usr/include/python2.7
culibraries := -lcublas


lib_paths   := $(culib_paths) $(cclib_paths) $(clib_paths)
libraries   := $(culibraries) $(cclibraries) $(clibraries)


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and Cuda files.

csources  := $(filter %.c ,$(sources))
ccsources := $(filter %.cc,$(sources))
cusources := $(filter %.cu,$(sources))

# This code generates a list of object files by replacing filename extensions

objects := $(patsubst %.c,%.o ,$(csources))  \
           $(patsubst %.cu,%.o,$(cusources)) \
           $(patsubst %.cc,%.o,$(ccsources))


#VPATH = ../cuda

#----- Build rules ------



$(target): $(objects) 
	$(gcc) -o $@ -shared $(lib_paths) $(libraries) $(objects)

%.o: %.cu
	$(nvcc) -c $^ $(cuflags) $(cuinc_paths) -o $@ 

%.cubin: %.cu
	$(nvcc) -cubin $(cuflags) $(cuinc_paths) $^

%.o: %.cc
	$(gpp) -c $^ $(ccflags) $(ccinc_paths) -o $@

%.o: %.c
	$(gcc) -c $^ $(cflags) $(cinc_paths) -o $@

clean:
	rm -f *.o $(target) makefile.*dep *.linkinfo test



##----- Dependency Generation -----
##
## If a particular set of sources is non-empty, then have rules for
## generating the necessary dep files.
##
#
#ccdep := ccdep.mk
#cdep  := cdep.mk
#cudep := cudep.mk
#
#
#depfiles =
#
#ifneq ($(ccsources),)
#
#depfiles += $(ccdep)
#$(ccdep): $(ccsources)
#	$(gpp) -MM $(ccsources) > $(ccdep)
#
#else
#
#$(ccdep):
#
#endif
#
#ifneq ($(cusources),)
#
#depfiles += $(cudep)
#$(cudep):
#	$(gpp) -MM -x c++ $(cusources) > $(cudep)
#
#else
#
#$(cudep):
#
#endif
#
#ifneq ($(csources),)
#
#depfiles += $(cdep)
#$(cdep): $(csources)
#	$(gcc) -MM -x c $(csources) > $(cdep)
#
#else
#
#$(cdep):
#
#endif
#
#.PHONY: dep
#dep: $(depfiles)
#
#
#ifneq ($(MAKECMDGOALS),dep)
# ifneq ($(MAKECMDGOALS),clean)
#  ifneq ($(ccsources),)
#   include $(ccdep)
#  endif
#  ifneq ($(cusources),)
#   include $(cudep)
#  endif
#  ifneq ($(csources),)
#   include $(cdep)
#  endif
# endif
#endif
