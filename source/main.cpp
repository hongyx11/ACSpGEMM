//  Project AC-SpGEMM
//  https://www.tugraz.at/institute/icg/research/team-steinberger/
//
//  Copyright (C) 2018 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Martin Winter - martin.winter (at) icg.tugraz.at
//              Daniel Mlakar - daniel.mlakar (at) icg.tugraz.at
//              Rhaleb Zayer - rzayer (at) mpi-inf.mpg.de
//              Hans-Peter Seidel - hpseidel (at) mpi-inf.mpg.de
//              Markus Steinberger - steinberger ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

/*!/------------------------------------------------------------------------------
 * Main.cpp
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

// Global includes
#include <fstream>
#include <iostream>
#include <spformat/COO.hpp>
#include <string>
#include <random>
#include <algorithm>
#include <tuple>
#include <cuda_runtime.h>

#include <spformat/spformat.hpp>
using namespace spformat;

// Local includes
#include "acspgemm/Multiply.h"
#include "acspgemm/Compare.h"




// CuSparse include
// #include "cusparse12/include/cuSparseMultiply.h"

// // Nsparse include
// #ifndef NONSPARSE
// #include "nsparse/include/nsparseMultiply.h"
// #endif

// // RMerge include
// #ifndef NORMERGE
// #include "RMerge/include/rmergeMultiply.h"
// #endif

// // BhSparse include
// #ifndef NOBHSPARSE
// #include"bhSparse/include/bhSparseMultiply.h"
// #endif
using IndexType = uint32_t;
using DataType = float;
unsigned int padding = 0;
template<typename T>
std::string typeext();
template<> 
std::string typeext<float>()
{
	return std::string("");
}
template<> 
std::string typeext<double>()
{
	return std::string("d_");
}

void printCheckMark()
{
	printf("\n        #\n       #\n      #\n #   #\n  # #\n   #\n\n");
}

void printCross()
{
	printf("\n #     # \n  #   #  \n   # #   \n    #    \n   # #   \n  #   #  \n #     # \n\n");
}

int main(int argc, char *argv[])
{
    std::cout << "########## ac-SpGEMM ##########" << std::endl;

    char  *filename;
    bool print_stats{ false };
    if (argc == 1)
    {
        std::cout << "Require filename of .mtx as first argument" << std::endl;
        return -1;
    }

    filename = argv[1];

    int device = 0;
    if (argc >= 3)
        device = std::stoi(argv[2]);

    bool testing = true;
    if(argc >= 4)
        testing = std::stoi(argv[3]) > 0 ? true : false;

    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

    // CSR matrices on the device
    CSR<IndexType,DataType> csr_mat, csr_T_mat, result_mat, test_mat;
    dCSR<IndexType,DataType> dcsr_mat, dcsr_T_mat, d_result_mat, d_result_mat_comp;//, d_nsparse_result_mat, d_rmerge_result_mat, d_bhSparse_result_mat;

    COO<IndexType,DataType> coo_mat;
    coo_mat.loadMTX(argv[1]);
    convert(csr_mat, coo_mat);
    // //try load csr file
    // std::string csr_name = std::string(argv[1]) + typeext<DataType>() + ".hicsr";
    // try
    // {
    //     std::cout << "trying to load csr file \"" << csr_name << "\"\n";
    //     csr_mat = loadCSR<DataType>(csr_name.c_str());
    //     std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
    // }
    // catch (std::exception& ex)
    // {
    //     std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
    //     try
    //     {
    //         std::cout << "trying to load mtx file \"" << argv[1] << "\"\n";
    //         COO<DataType> coo_mat = loadMTX<DataType>(argv[1]);
    //         convert(csr_mat, coo_mat);
    //         std::cout << "succesfully loaded and converted: \"" << csr_name << "\"\n";
    //     }
    //     catch (std::exception& ex)
    //     {
    //         std::cout << ex.what() << std::endl;
    //         return -1;
    //     }
    //     try
    //     {
    //         std::cout << "write csr file for future use\n";
    //         storeCSR(csr_mat, csr_name.c_str());
    //     }
    //     catch (std::exception& ex)
    //     {
    //         std::cout << ex.what() << std::endl;
    //     }
    // }

    // Convert host csr to device csr
    convert(dcsr_mat, csr_mat, padding);

    // Cusparse object
    // cuSPARSE::CuSparseTest<DataType> cusparse;

    // bool transpose = (dcsr_mat.rows != dcsr_mat.cols);
    // if (transpose)
    // {
    // 	std::cout << "Matrix not square (" << dcsr_mat.rows << "x" << dcsr_mat.cols << ") - Calculate Transpose!\n";
    // 	/*ACSpGEMM::Transpose(dcsr_mat, dcsr_T_mat);*/
    // 	cusparse.Transpose(dcsr_mat, dcsr_T_mat);
    // 	convert(csr_T_mat, dcsr_T_mat, padding);
    // }

    bool transpose = false;

    printf("Input Matrix A: (%zu x %zu) - NNZ: %zu\n", dcsr_mat.nrows_, dcsr_mat.ncols_, dcsr_mat.nnz_);
    if(transpose)
        printf("Input Matrix B: (%zu x %zu) - NNZ: %zu\n", dcsr_T_mat.nrows_, dcsr_T_mat.ncols_, dcsr_T_mat.nnz_);

    const int Threads = 256;
    const int BlocksPerMP = 3;
    const int NNZPerThread = 2;
    const int InputElementsPerThreads = 4;
    const int RetainElementsPerThreads = 4;
    const int MaxChunksToMerge = 16;
    const int MaxChunksGeneralizedMerge = 512; // MAX: 865
    const int MergePathOptions = 8;

    GPUMatrixMatrixMultiplyTraits DefaultTraits(
        Threads, 
        BlocksPerMP, 
        NNZPerThread, 
        InputElementsPerThreads, 
        RetainElementsPerThreads, 
        MaxChunksToMerge, 
        MaxChunksGeneralizedMerge, 
        MergePathOptions); // DefaultTraits(128, 2, 4, 1, 8, 128, 8);
    const bool Debug_Mode = true;
    bool checkBitStability{false};
    DefaultTraits.preferLoadBalancing = true;
    ExecutionStats stats, warmupstats, output_stats;
    stats.measure_all = false;
    output_stats.measure_all = false;

    uint32_t warmupiterations = testing ? checkBitStability ? 1 : 0 : 5;
    uint32_t iterations = testing ? 1 : 10;

    try {
        // Warmup iterations for multiplication
        for (uint32_t i = 0; i < warmupiterations; ++i)
        {
            warmupstats.reset();
            if(testing) std::cerr<<"warmup iter" << i << std::endl;
            ACSpGEMM::Multiply<IndexType,DataType>(dcsr_mat, transpose ? dcsr_T_mat : dcsr_mat, d_result_mat_comp, DefaultTraits, warmupstats, Debug_Mode);
        }

        // Multiplication
        for (uint32_t i = 0; i < iterations; ++i)
        {
            if (testing) std::cerr << "Iteration: " << i + 1 << "\n";
            std::cerr << "Iteration: " << i + 1 << "\n";
            ACSpGEMM::Multiply<IndexType,DataType>(dcsr_mat, transpose ? dcsr_T_mat : dcsr_mat, d_result_mat, DefaultTraits, stats, Debug_Mode);
            if(checkBitStability)
            {
                if (!ACSpGEMM::Compare<IndexType,DataType>(d_result_mat_comp, d_result_mat, true))
                {
                    printf("NOT Bit-Identical\n");
                    printCross();
                    exit(-1);
                }
                else
                {
                    printf("Bit-Identical\n");
                    printCheckMark();
                }
            }
            output_stats += stats;
            stats.reset();
        }
    }
    catch (std::exception& ex)
    {
        std::cout << "Caught exception in acSpGEMM\n";
        std::cout << ex.what() << std::endl;
    }

    // output_stats.normalize();
    // std::cout << output_stats;
    // std::cout << "-----------------------------------------------\n";


    // if(checkBitStability)
    //     return 0;

    // // cuSparse Multiplication
    // dCSR<IndexType,DataType> d_cusparse_result_mat;
    // bool test_data = false;
    // uint32_t cusparse_nnz;
    // float cusparse_performance = 0.0f;
    // try {
    //     for (uint32_t i = 0; i < warmupiterations; ++i)
    //     {
    //         std::cerr << "warmupIteration: " << i + 1 << "\n";
    //         cusparse.Multiply(dcsr_mat, transpose ? dcsr_T_mat : dcsr_mat, d_cusparse_result_mat, cusparse_nnz);
    //     }
    //     for (uint32_t i = 0; i < iterations; ++i)
    //     {
    //         std::cerr << "Iteration: " << i + 1 << "\n";
    //         cudaDeviceSynchronize();
    //         cusparse_performance += cusparse.Multiply(dcsr_mat, transpose ? dcsr_T_mat : dcsr_mat, d_cusparse_result_mat, cusparse_nnz);
    //         cudaDeviceSynchronize();
    //     }
    // }
    // catch (std::exception&)
    // {
    //     std::cout << "Caught exception in cusparse\n";
    //     cusparse_performance = -1.0f;
    // }

    // std::cout << "Overall Duration (cusparse): " << cusparse_performance / iterations << " ms\n";
    // std::cout << "-----------------------------------------------\n";

    // if (d_cusparse_result_mat.nnz_ != d_result_mat.nnz_)
    // {
    //     std::cout << "NNZ (cuSparse " << d_cusparse_result_mat.nnz_ << "|" << d_result_mat.nnz_ << " ac-SpGEMM) NOT identical!\n";
    //     printCross();
    // }
    // if (ACSpGEMM::Compare<IndexType,DataType>(d_cusparse_result_mat, d_result_mat, test_data))
    // {
    //     printf("(cuSPARSE %zu | ac-SpGEMM %zu) IDENTICAL\n", d_cusparse_result_mat.nnz_, d_result_mat.nnz_);
    //     printCheckMark();
    // }
    // else
    // {
    //     printf("(cuSPARSE %zu | ac-SpGEMM %zu) NOT IDENTICAL | Missing: %d\n", d_cusparse_result_mat.nnz_, d_result_mat.nnz_, (int)d_cusparse_result_mat.nnz_ - (int)d_result_mat.nnz_);
    //     printCross();
    // }
    // if (testing)
    // {
    //     // Compare matrices cuSparse / acSpGEMM
    //     if (d_cusparse_result_mat.nnz_ != d_result_mat.nnz_)
    //     {
    //         std::cout << "NNZ (cuSparse " << d_cusparse_result_mat.nnz_ << "|" << d_result_mat.nnz_ << " ac-SpGEMM) NOT identical!\n";
    //         printCross();
    //     }
    //     if (ACSpGEMM::Compare<IndexType,DataType>(d_cusparse_result_mat, d_result_mat, test_data))
    //     {
    //         printf("(cuSPARSE %zu | ac-SpGEMM %zu) IDENTICAL\n", d_cusparse_result_mat.nnz_, d_result_mat.nnz_);
    //         printCheckMark();
    //     }
    //     else
    //     {
    //         printf("(cuSPARSE %zu | ac-SpGEMM %zu) NOT IDENTICAL | Missing: %d\n", d_cusparse_result_mat.nnz_, d_result_mat.nnz_, (int)d_cusparse_result_mat.nnz_ - (int)d_result_mat.nnz_);
    //         printCross();
    //     }

    //     /*convert(result_mat, d_result_mat);
    //     convert(test_mat, d_cusparse_result_mat);
    //     uint32_t number_rows_to_visit{ 1 };
    //     for (int row = 0; row < test_mat.rows - 1, number_rows_to_visit > 0; ++row)
    //     {
    //         uint32_t interesting_row = row;
    //         uint32_t ref_offset = test_mat.row_offsets[interesting_row];
    //         uint32_t comp_offset = result_mat.row_offsets[interesting_row];
    //         uint32_t ref_number_entries = test_mat.row_offsets[interesting_row + 1] - ref_offset;
    //         uint32_t comp_number_entries = result_mat.row_offsets[interesting_row + 1] - comp_offset;

    //         if (ref_number_entries != comp_number_entries)
    //         {
    //             printf("Row: %u -- Entries not identical - cusparse: %u | %u acspgemm\n", row, ref_number_entries, comp_number_entries);
    //             printf("cuSparse\n");
    //             for (uint32_t i = ref_offset; i < ref_offset + ref_number_entries; ++i)
    //             {
    //                 printf("%u | ", test_mat.col_ids[i]);
    //             }
    //             printf("\n");
    //             printf("acspgemm\n");
    //             for (uint32_t i = comp_offset; i < comp_offset + comp_number_entries; ++i)
    //             {
    //                 printf("%u | ", result_mat.col_ids[i]);
    //             }
    //             printf("\n---------------------------------------------------------------------------\n");
    //             --number_rows_to_visit;
    //         }
    //     }*/
    // }



    return 0;
}