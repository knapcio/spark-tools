// SPDX-License-Identifier: MIT
// Copyright (c) 2026 knapcio
// CPU-only use of NVIDIA's installed occupancy calculator; no CUDA runtime calls.
#include <cuda_occupancy.h>
#include <cstdlib>
#include <iostream>
int main(int argc,char** argv) {
  if(argc!=6)return 2;
  cudaOccDeviceProp p;
  p.computeMajor=12;p.computeMinor=1;p.maxThreadsPerBlock=1024;
  p.maxThreadsPerMultiprocessor=1536;p.regsPerBlock=65536;p.regsPerMultiprocessor=65536;
  p.warpSize=32;p.sharedMemPerBlock=49152;p.sharedMemPerMultiprocessor=102400;
  p.numSms=48;p.sharedMemPerBlockOptin=101376;p.reservedSharedMemPerBlock=1024;
  cudaOccFuncAttributes a;
  a.maxThreadsPerBlock=1024;a.numRegs=std::atoi(argv[1]);a.sharedSizeBytes=std::strtoull(argv[3],nullptr,10);
  a.shmemLimitConfig=FUNC_SHMEM_LIMIT_OPTIN;a.maxDynamicSharedSizeBytes=101376-a.sharedSizeBytes;
  a.numBlockBarriers=std::atoi(argv[5]);
  cudaOccDeviceState s; s.carveoutConfig=100;
  cudaOccResult r{};
  int threads=std::atoi(argv[2]);size_t dynamic=std::strtoull(argv[4],nullptr,10);
  auto e=cudaOccMaxActiveBlocksPerMultiprocessor(&r,&p,&a,&s,threads,dynamic);
  if(e!=CUDA_OCC_SUCCESS){std::cerr << "cudaOcc error " << int(e);return 3;}
  std::cout << "{\"status\":\"modeled\",\"method\":\"CUDA13 cuda_occupancy.h CPU calculator\","
    << "\"threads\":" << threads << ",\"dynamic_shared\":" << dynamic
    << ",\"active_blocks_per_sm\":" << r.activeBlocksPerMultiprocessor
    << ",\"active_blocks_total\":" << r.activeBlocksPerMultiprocessor*48
    << ",\"warp_occupancy_pct\":" << r.activeBlocksPerMultiprocessor*((threads+31)/32)*100.0/48
    << ",\"allocated_registers_per_CTA\":" << r.allocatedRegistersPerBlock
    << ",\"allocated_shared_per_CTA\":" << r.allocatedSharedMemPerBlock
    << ",\"limits\":{\"registers\":" << r.blockLimitRegs << ",\"shared\":" << r.blockLimitSharedMem
    << ",\"warps\":" << r.blockLimitWarps << ",\"blocks\":" << r.blockLimitBlocks
    << ",\"barriers\":" << r.blockLimitBarriers << "},\"limiting_bitmask\":" << r.limitingFactors << "}\n";
}
