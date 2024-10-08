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

#pragma once
#include <cstdint>
#include <cstddef>


struct GeneralSchedulingTraits
{
	static const bool MultiGPU = false;

	bool preferLoadBalancing;
	size_t cpu_threads;
	int device;

	GeneralSchedulingTraits() : cpu_threads(8), device(0), preferLoadBalancing(true) { }
};

struct AVX2SchedulingTratis : public GeneralSchedulingTraits{};

struct DefaultSchedulingTraits : public GeneralSchedulingTraits {};

struct GPUMatrixMatrixMultiplyTraits : public GeneralSchedulingTraits
{
	const int Threads;
	const int BlocksPerMp;
	const int NNZPerThread;
	const int InputElementsPerThreads;
	const int RetainElementsPerThreads;
	const int MaxChunksToMerge;
	const int MaxChunksGeneralizedMerge;
	const int MergePathOptions;


	GPUMatrixMatrixMultiplyTraits(
	   const int Threads = 256,
	   const int BlocksPerMp = 3,
	   const int NNZPerThread = 2,
	   const int InputElementsPerThreads = 4,
	   const int RetainElementsPerThreads = 4,
	   const int MaxChunksToMerge = 16,
	   const int MaxChunksGeneralizedMerge = 256,
	   const int MergePathOptions = 8) :
		Threads(Threads),
		BlocksPerMp(BlocksPerMp),
		NNZPerThread(NNZPerThread),
		InputElementsPerThreads(InputElementsPerThreads),
		RetainElementsPerThreads(RetainElementsPerThreads),
		MaxChunksToMerge(MaxChunksToMerge),
		MaxChunksGeneralizedMerge(MaxChunksGeneralizedMerge),
		MergePathOptions(MergePathOptions)
	{}
};