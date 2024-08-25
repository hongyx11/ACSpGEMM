/*!/------------------------------------------------------------------------------
 * cuSparseMultiply.cu
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
 */

#include "cusparse12/include/cuSparseMultiply.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

#define CHECK_CUSPARSE_NORET(func)                                             \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
    }                                                                          \
  }

namespace cuSPARSE {

template <typename DataType>
float CuSparseTest<DataType>::Multiply(const dCSR<DataType> &A,
                                       const dCSR<DataType> &B,
                                       dCSR<DataType> &c,
                                       uint32_t &cusparse_nnz) {
    float duration;
    int m, n, k;
    m = A.rows;
    n = B.cols;
    k = A.cols;
    c.reset();

    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matB;
    cusparseSpMatDescr_t matC;
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ############################
    cudaEventRecord(start);
    // ############################

    // Allocate memory for row indices
    cudaMalloc(&(c.row_offsets), sizeof(uint32_t) * (A.rows + 1));

    // CUSPARSE APIs
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, k, A.nnz, A.row_offsets, A.col_ids, A.data, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matB, k, n, B.nnz, B.row_offsets, B.col_ids, B.data, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, m, n, 0, c.row_offsets, NULL, NULL,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    DataType alpha = 1.0;
    DataType beta = 0.0;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
    
    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL))
    cudaMalloc((void **)&dBuffer1, bufferSize1);
    
    // inspect the matrices A and B to understand the memory requirement for the
    // next step
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1))
    
    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_compute(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL))
    cudaMalloc((void **)&dBuffer2, bufferSize2);

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2))
    
    // get matrix C non-zero entries C_nnz1
    int64_t cnrow, cncolmn, cnnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &cnrow, &cncolmn, &cnnz))
    c.rows = cnrow;
    c.cols = cncolmn;
    c.nnz = cnnz;
    
    // allocate matrix C
    cudaMalloc((void **)&c.col_ids, c.nnz * sizeof(uint32_t));
    cudaMalloc((void **)&c.data, c.nnz * sizeof(DataType));
    cusparse_nnz = c.nnz;

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, c.row_offsets, c.col_ids, c.data));

    // copy the final products to the matrix C
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB,
                                        &beta, matC, computeType,
                                        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    cusparseDestroy(handle);
    // ############################
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // ############################

    cudaEventElapsedTime(&duration, start, stop);
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cudaDeviceSynchronize();
    return duration;
}

template float CuSparseTest<float>::Multiply(const dCSR<float> &A,
                                             const dCSR<float> &B,
                                             dCSR<float> &matOut,
                                             uint32_t &cusparse_nnz);
template float CuSparseTest<double>::Multiply(const dCSR<double> &A,
                                              const dCSR<double> &B,
                                              dCSR<double> &matOut,
                                              uint32_t &cusparse_nnz);

template <typename DataType>
void CuSparseTest<DataType>::Transpose(const dCSR<DataType> &A,
                                       dCSR<DataType> &AT) {}

} // namespace cuSPARSE