#!/bin/bash

libtool --mode=link gcc -static -o libnmf.a build/temp/matrix.o build/temp/run_nmf.o -lmkl_em64
