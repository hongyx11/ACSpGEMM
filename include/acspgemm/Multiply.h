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
* Multiply.h
*
* ac-SpGEMM
*
* Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
*------------------------------------------------------------------------------
*/
#pragma once

// #include "dCSR.h"
#include <spformat/spformat.hpp>
#include "execution_stats.h"
#include "default_scheduling_traits.h"
#include <cuda.h>
#include <cuda_runtime.h>



namespace ACSpGEMM {

    template <
    typename IndexType = uint32_t, 
    typename DataType
    >
    void Multiply(
        const spformat::dCSR<IndexType, DataType>& A, 
        const spformat::dCSR<IndexType,DataType>& B, 
        spformat::dCSR<IndexType,DataType>& matOut, 
        const GPUMatrixMatrixMultiplyTraits& scheduling_traits, 
        ExecutionStats& exec_stats, bool Debug_Mode = false);

    template <
    typename IndexType = uint32_t,
    typename DataType, 
    uint32_t threads, 
    uint32_t blocks_per_mp, 
    uint32_t nnz_per_thread, 
    uint32_t input_elements_per_thread, 
    uint32_t retain_elements_per_thread, 
    uint32_t merge_max_chunks, 
    uint32_t generalized_merge_max_path_options, 
    uint32_t merge_max_path_options, 
    bool DEBUG_MODE=false
    >
    void MultiplyImplementation(
        const spformat::dCSR<IndexType, DataType>& A, 
        const spformat::dCSR<IndexType, DataType>& B, 
        spformat::dCSR<IndexType, DataType>& matOut, 
        const GPUMatrixMatrixMultiplyTraits& scheduling_traits, 
        ExecutionStats& exec_stats);


}


