CU_APPS=cusparse-matrix-matrix access-ordering cublas-matrix-matrix \
		cublas-matrix-matrix-async cufft-inverse cusparse-matrix-matrix-double \
		example
C_APPS=

ifeq ($(DEBUG),ON)
	FLAG=-DDEBUG
endif

all: ${C_APPS} ${CU_APPS}

cusparse-matrix-matrix: cusparse-matrix-matrix.cu
	nvcc -O3 -arch=sm_75 $(FLAG) -o cusparse-matrix-matrix cusparse-matrix-matrix.cu -lcusparse

cublas-matrix-matrix: cublas-matrix-matrix.cu
	nvcc -O3 -arch=sm_75 -o cublas-matrix-matrix cublas-matrix-matrix.cu -lcublas

cublas-matrix-matrix-async: cublas-matrix-matrix-async.cu
	nvcc -O3 -arch=sm_75 -o cublas-matrix-matrix-async cublas-matrix-matrix-async.cu -lcublas

cufft-inverse: cufft-inverse.cu
	nvcc -O3 -arch=sm_75 -o cufft-inverse cufft-inverse.cu -lcufft -lm

cusparse-matrix-matrix-double: cusparse-matrix-matrix-double.cu
	nvcc -O3 -arch=sm_75 -o cusparse-matrix-matrix-double cusparse-matrix-matrix-double.cu -lcusparse

example: example.cu
	nvcc -O3 -arch=sm_75 -o example example.cu -lcusparse


%: %.cu
	nvcc -O3 -arch=sm_75 -o $@ $<
%: %.c
	gcc -O3 -std=c99 -lm -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
