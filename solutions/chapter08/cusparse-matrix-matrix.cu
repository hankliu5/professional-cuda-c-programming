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

            if (r % 2 > 0)
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
    float *A, *dA;
    float *B, *dB;
    int i, j, error = 0;
    float *Bt;
    float *cpu_C;
    float *C, *dC;
    int M, N;
    int *dANnzPerRow;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;

    // transfer by CPU
    float *CsrValA_cpu;
    int *CsrRowPtrA_cpu;
    int *CsrColIndA_cpu;

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
    
    cpu_C = (float *)malloc(sizeof(float) * M * M);
    Bt = (float *)malloc(sizeof(float) * N * M);
    transpose(B, Bt, M, N);

#ifdef DEBUG
    printf("A:\n");
    print_matrix(A, M, N);
    printf("B:\n");
    print_matrix(B, N, M);
    printf("Bt:\n");
    print_matrix(Bt, N, M);
#endif

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
        
    CsrValA_cpu = (float *) malloc(sizeof(float) * totalANnz);
    CsrRowPtrA_cpu = (int *) malloc(sizeof(int) * (M + 1));
    CsrColIndA_cpu = (int *) malloc(sizeof(int) * totalANnz);

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    printf("M: %d, N: %d\n", M, N);
    start = clock();
    dense2csr(A, M, N, CsrValA_cpu, CsrRowPtrA_cpu, CsrColIndA_cpu);
    diff = clock() - start;
    printf("dense2csr: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);

    CHECK(cudaMemcpy(dCsrValA, CsrValA_cpu, sizeof(float) * totalANnz, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dCsrRowPtrA, CsrRowPtrA_cpu, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dCsrColIndA, CsrColIndA_cpu, sizeof(int) * totalANnz, cudaMemcpyHostToDevice));

    start = clock();
    densemm(A, Bt, cpu_C, M, N);
    diff = clock() - start;
    printf("cpu naive densemm: %ld msec\n", diff * 1000 / CLOCKS_PER_SEC);

#ifdef DEBUG
    printf("C = A * Bt:\n");
    print_matrix(cpu_C, M, N);
#endif    
    
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

    for (j = 0; j < M; j++){
        for (i = 0; i < M; i++) {
            if (abs(C[i+M*j] - cpu_C[i*M+j]) > 1) {
                printf("C[%d,%d]=%f, cpu_C[%d,%d]=%f\n",i,j,C[i+M*j],i,j,cpu_C[i*M+j]);
                error++;
            }
        }
    }
    if (error) {
        printf("Result incorrect\n");
    }
    else {
        printf("pass\n");
    }
    free(Bt);
    free(cpu_C);

    free(A);
    free(B);
    free(C);

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
