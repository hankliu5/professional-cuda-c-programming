#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda.h>
#include <time.h>


/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
 */


/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float *curr = A + (j * M + i);

            // if (r % 3 > 0)
            // {
            //     *curr = 0.0f;
            // }
            // else
            // {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            // }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

void print_matrix(float *M, int nrows, int ncols)
{
    int row, col;

    for (row = 0; row < nrows; row++)
    {
        for (col = 0; col < ncols; col++)
        {
            printf("%2.2f ", M[row * ncols + col]);
        }
        printf("\n");
    }
}


void dense2csr(float *M, int nrows, int ncols, float *CsrVal, int *CsrRowPtr, int *CsrColInd) {
    int i, j, num_of_NNZ = 0, count = 0;
    CsrRowPtr[0] = 0;
    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ncols; j++, count++) {
            if (M[count] != 0.0f) {
                CsrVal[num_of_NNZ] = M[count];
                CsrColInd[num_of_NNZ] = j;
                num_of_NNZ++;
            }
        }
        CsrRowPtr[i+1] = num_of_NNZ;
    }
    printf("counts: %d\n", count);
}

void densemm(float *A, float *B, float *C, int nrows, int ncols) {
    int i, j, k;
    for(i = 0; i < nrows; i++)    
    {    
        for(j = 0; j < ncols; j++)    
        {    
            C[i * ncols + j] = 0;    
            for(k = 0; k < ncols; k++)    
            {    
                C[i * ncols + j] += A[i * ncols + k] * B[k * ncols + j];    
            }    
        }    
    }   
}

void transpose(float *A, float *At, int nrows, int ncols) {
    int i, j;
    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ncols; j++) {
            At[j * ncols + i] = A[i * ncols + j];
        }
    }
}


int main(int argc, char **argv)
{
    float *A, *dA, *At;
    float *B, *dB, *Bt;
    float *C, *dC;
    int M, N;
    int *dANnzPerRow;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;

    // transfer by CUDA
    float *CsrValA_cuda;
    int *CsrRowPtrA_cuda;
    int *CsrColIndA_cuda;

    // transfer by CPU
    float *CsrValA_cpu;
    int *CsrRowPtrA_cpu;
    int *CsrColIndA_cpu;

    int i;

    int totalANnz;
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t Adescr = 0;
    clock_t start, diff;

    if (argc != 3) {
        printf("usage: %s <# of rows> <# of cols>\n", argv[0]);
        exit(1);
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    
    // Generate input
    srand(9384);
    start = clock();
    int trueANnz = generate_random_dense_matrix(M, N, &A);
    int trueBNnz = generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);
    diff = clock() - start;
    printf("generate dataset: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);

    printf("A:\n");
    print_matrix(A, M, N);
    printf("B:\n");
    print_matrix(B, N, M);

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Allocate device memory for vectors and the dense form of the matrix A
    start = clock();
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));
    CHECK(cudaMalloc((void **)&dANnzPerRow, sizeof(int) * M));
    diff = clock() - start;
    printf("cudaMalloc dense matrix: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);


    // Construct a descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr(&Adescr));
    CHECK_CUSPARSE(cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO));

    // Transfer the input vectors and dense matrix A to the device
    start = clock();
    CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * N * M, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0x00, sizeof(float) * M * M));
    diff = clock() - start;
    printf("cudaMemcpy dense matrix: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);

    // Compute the number of non-zero elements in A
    start = clock();
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, Adescr,
                                dA, M, dANnzPerRow, &totalANnz));
    diff = clock() - start;
    printf("cusparseSnnz: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);
                            
    
    if (totalANnz != trueANnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueANnz, totalANnz);
        return 1;
    }

    printf("totalANnz: %d\n", totalANnz);
    printf("sparsity: %f\n", 1.0f - (float) totalANnz / (M * N));

    // Allocate device memory to store the sparse CSR representation of A
    start = clock();
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalANnz));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalANnz));
    diff = clock() - start;
    printf("cudaMalloc CSR matrix: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);


    // Convert A from a dense formatting to a CSR formatting, using the GPU
    start = clock();
    CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, Adescr, dA, M, dANnzPerRow,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));
    diff = clock() - start;
    printf("cusparseSdense2csr: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);
    
    CsrValA_cuda = (float *) malloc(sizeof(float) * totalANnz);
    CsrRowPtrA_cuda = (int *) malloc(sizeof(int) * (M + 1));
    CsrColIndA_cuda = (int *) malloc(sizeof(int) * totalANnz);
    
    CHECK(cudaMemcpy(CsrValA_cuda, dCsrValA, sizeof(float) * totalANnz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(CsrRowPtrA_cuda, dCsrRowPtrA, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(CsrColIndA_cuda, dCsrColIndA, sizeof(int) * totalANnz, cudaMemcpyDeviceToHost));

    CsrValA_cpu = (float *) malloc(sizeof(float) * totalANnz);
    CsrRowPtrA_cpu = (int *) malloc(sizeof(int) * (M + 1));
    CsrColIndA_cpu = (int *) malloc(sizeof(int) * totalANnz);

    Bt = (float *)malloc(sizeof(float) * N * M);

    transpose(B, Bt, M, N);
    printf("Bt:\n");
    print_matrix(Bt, N, M);

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    printf("M: %d, N: %d\n", M, N);
    start = clock();
    dense2csr(A, M, N, CsrValA_cpu, CsrRowPtrA_cpu, CsrColIndA_cpu);
    diff = clock() - start;
    printf("dense2csr: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);

    densemm(A, B, C, M, N);
    printf("C = A * B:\n");
    print_matrix(C, M, N);

    densemm(A, Bt, C, M, N);
    printf("C = A * Bt:\n");
    print_matrix(C, M, N);

    printf("\n");
    for (i = 0; i < totalANnz; i++) {
        printf("%f %f\n", CsrValA_cuda[i], CsrValA_cpu[i]);
    }

    for (i = 0; i < (M + 1); i++) {
        printf("%d %d\n", CsrRowPtrA_cuda[i], CsrRowPtrA_cpu[i]);
    }

    for (i = 0; i < totalANnz; i++) {
        printf("%d %d\n", CsrColIndA_cuda[i], CsrColIndA_cpu[i]);
    }
    
    
    // Perform matrix-matrix multiplication with the CSR-formatted matrix A
    start = clock();
    CHECK_CUSPARSE(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M,
                                  M, N, totalANnz, &alpha, Adescr, dCsrValA,
                                  dCsrRowPtrA, dCsrColIndA, dB, N, &beta, dC,
                                  M));
    diff = clock() - start;
    printf("cusparseScsrmm: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);
                              
    // Copy the result vector back to the host
    start = clock();
    CHECK(cudaMemcpy(C, dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost));
    diff = clock() - start;
    printf("cudaMemcpy result dense matrix: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);

    printf("C:\n");
    printf("Final results:\n");
    for (int j=0; j<N; j++){
        for (int i=0; i<M; i++){
            printf("C[%d,%d]=%f\n",i,j,C[i+M*j]);
        }
    }

    free(A);
    free(B);
    free(C);
    free(Bt);

    free(CsrValA_cuda);
    free(CsrRowPtrA_cuda);
    free(CsrColIndA_cuda);

    free(CsrValA_cpu);
    free(CsrRowPtrA_cpu);
    free(CsrColIndA_cpu);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaFree(dANnzPerRow));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(Adescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}
