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
 * M = # of rows
 * N = # of columns
 */
// int M = 16384;
// int N = 16384;

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

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

void print_partial_matrix(float *M, int nrows, int ncols, int max_row,
        int max_col)
{
    int row, col;

    for (row = 0; row < max_row; row++)
    {
        for (col = 0; col < max_col; col++)
        {
            printf("%2.2f ", M[row * ncols + col]);
        }
        printf("...\n");
    }
    printf("...\n");
}

int main(int argc, char **argv)
{
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    int M, N;
    int *dANnzPerRow;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    int totalANnz;
    float alpha = 3.0f;
    float beta = 4.0f;
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
    print_partial_matrix(A, M, N, 10, 10);
    printf("B:\n");
    print_partial_matrix(B, N, M, 10, 10);

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
    printf("sparsity: %f\n", (float) totalANnz / (M * N));

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
    print_partial_matrix(C, M, M, 10, 10);

    free(A);
    free(B);
    free(C);

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
