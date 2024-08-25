/*!/------------------------------------------------------------------------------
* Multiply.h
*
* cuSparse Multiplication functionality
*
* Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
*------------------------------------------------------------------------------
*/

#pragma once

#include "dCSR.h"
#include <cusparse.h>
#include <iostream>
#include <string>

namespace cuSPARSE {

    template <typename DataType>
    class CuSparseTest
    {

    public:
        CuSparseTest(){};
        ~CuSparseTest(){};

        // Multiply two CSR matrices
        float Multiply(const dCSR<DataType>& A, const dCSR<DataType>& B, dCSR<DataType>& matOut, uint32_t& cusparse_nnz);

        void Transpose(const dCSR<DataType>& A, dCSR<DataType>& AT);

        void checkCuSparseError(cusparseStatus_t status, std::string errorMsg)
        {
            if (status != CUSPARSE_STATUS_SUCCESS) {
                std::cout << "CuSparse error: " << errorMsg << std::endl;
                throw std::exception();
            }
        }

        cusparseStatus_t CUSPARSEAPI cusparseMultiply(cusparseHandle_t handle,
            cusparseOperation_t transA,
            cusparseOperation_t transB,
            int m,
            int n,
            int k,
            const cusparseMatDescr_t descrA,
            int nnzA,
            const DataType *csrSortedValA,
            const int *csrSortedRowPtrA,
            const int *csrSortedColIndA,
            const cusparseMatDescr_t descrB,
            int nnzB,
            const DataType *csrSortedValB,
            const int *csrSortedRowPtrB,
            const int *csrSortedColIndB,
            const cusparseMatDescr_t descrC,
            DataType *csrSortedValC,
            const int *csrSortedRowPtrC,
            int *csrSortedColIndC);

        cusparseStatus_t CUSPARSEAPI cusparseTranspose(cusparseHandle_t handle,
            int m,
            int n,
            int nnz,
            const DataType  *csrSortedVal,
            const int *csrSortedRowPtr,
            const int *csrSortedColInd,
            DataType *cscSortedVal,
            int *cscSortedRowInd,
            int *cscSortedColPtr,
            cusparseAction_t copyValues,
            cusparseIndexBase_t idxBase);		
    };
}