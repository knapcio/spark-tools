// SPDX-License-Identifier: Apache-2.0
// Queries residency APIs only. No kernel launches or device allocations.
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include "kernels.inc"

static std::string quoted(const char* value) {
  std::string result = "\"";
  for (const char* p=value; *p; ++p) {
    if (*p=='\\' || *p=='"') result += '\\';
    if (static_cast<unsigned char>(*p)>=32) result += *p;
  }
  return result + "\"";
}

int main(int argc, char** argv) {
  if (argc != 9) return 2;
  int device=std::atoi(argv[1]), variant=std::atoi(argv[2]);
  int threads=std::atoi(argv[3]), optin=std::atoi(argv[5]);
  size_t dynamic=std::strtoull(argv[4],nullptr,10);
  int carveout=std::atoi(argv[6]), flags=std::atoi(argv[7]), barriers=std::atoi(argv[8]);
  if (variant<0 || variant>=int(sizeof(fixture_functions)/sizeof(fixture_functions[0]))) return 2;
  cudaError_t setup=cudaSetDevice(device);
  if (setup!=cudaSuccess) {
    std::cout << "{\"setup_error\":" << int(setup) << ",\"error\":" << quoted(cudaGetErrorName(setup)) << "}\n";
    return 0;
  }
  cudaDeviceProp p{};
  setup=cudaGetDeviceProperties(&p,device);
  if (setup!=cudaSuccess || p.major!=12 || p.minor!=1) {
    std::cout << "{\"setup_error\":" << int(setup) << ",\"unsupported_device\":true,"
              << "\"compute_major\":" << p.major << ",\"compute_minor\":" << p.minor << "}\n";
    return 0;
  }
  const void* function=fixture_functions[variant];
  cudaError_t shared=cudaSuccess;
  if (optin>=0) shared=cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,optin);
  cudaError_t carve=cudaFuncSetAttribute(function,cudaFuncAttributePreferredSharedMemoryCarveout,carveout);
  cudaFuncAttributes a{};
  cudaError_t attributes=cudaFuncGetAttributes(&a,function);
  int runtime_blocks=-1;
  cudaError_t runtime=flags==0
    ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&runtime_blocks,function,threads,dynamic)
    : cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&runtime_blocks,function,threads,dynamic,flags);
  cudaOccDeviceProp header_device(p);
  cudaOccFuncAttributes header_attributes(a);
  header_attributes.numBlockBarriers=barriers;
  header_attributes.partitionedGCConfig=a.cacheModeCA
    ? (flags==cudaOccupancyDisableCachingOverride ? PARTITIONED_GC_ON_STRICT : PARTITIONED_GC_ON)
    : PARTITIONED_GC_OFF;
  cudaOccDeviceState state;
  state.carveoutConfig=a.preferredShmemCarveout;
  cudaOccResult result{};
  cudaOccError header=cudaOccMaxActiveBlocksPerMultiprocessor(&result,&header_device,&header_attributes,&state,threads,dynamic);
  int driver=0, toolkit=0;
  cudaError_t driver_status=cudaDriverGetVersion(&driver), toolkit_status=cudaRuntimeGetVersion(&toolkit);
  std::cout << "{\"device\":{\"ordinal\":" << device << ",\"name\":" << quoted(p.name)
    << ",\"major\":" << p.major << ",\"minor\":" << p.minor
    << ",\"maxThreadsPerBlock\":" << p.maxThreadsPerBlock
    << ",\"maxThreadsPerMultiProcessor\":" << p.maxThreadsPerMultiProcessor
    << ",\"regsPerBlock\":" << p.regsPerBlock << ",\"regsPerMultiprocessor\":" << p.regsPerMultiprocessor
    << ",\"warpSize\":" << p.warpSize << ",\"multiProcessorCount\":" << p.multiProcessorCount
    << ",\"maxBlocksPerMultiProcessor\":" << p.maxBlocksPerMultiProcessor
    << ",\"sharedMemPerBlock\":" << p.sharedMemPerBlock
    << ",\"sharedMemPerMultiprocessor\":" << p.sharedMemPerMultiprocessor
    << ",\"sharedMemPerBlockOptin\":" << p.sharedMemPerBlockOptin
    << ",\"reservedSharedMemPerBlock\":" << p.reservedSharedMemPerBlock << "},"
    << "\"attributes\":{\"maxThreadsPerBlock\":" << a.maxThreadsPerBlock << ",\"numRegs\":" << a.numRegs
    << ",\"sharedSizeBytes\":" << a.sharedSizeBytes << ",\"localSizeBytes\":" << a.localSizeBytes
    << ",\"maxDynamicSharedSizeBytes\":" << a.maxDynamicSharedSizeBytes
    << ",\"preferredShmemCarveout\":" << a.preferredShmemCarveout
    << ",\"cacheModeCA\":" << a.cacheModeCA << ",\"binaryVersion\":" << a.binaryVersion
    << ",\"ptxVersion\":" << a.ptxVersion << "},"
    << "\"set_dynamic_error\":" << int(shared) << ",\"set_carveout_error\":" << int(carve)
    << ",\"attributes_error\":" << int(attributes)
    << ",\"runtime_error\":" << int(runtime) << ",\"runtime_blocks_per_sm\":" << runtime_blocks
    << ",\"header_error\":" << int(header) << ",\"header_blocks_per_sm\":" << result.activeBlocksPerMultiprocessor
    << ",\"header_barriers\":" << header_attributes.numBlockBarriers
    << ",\"header_shmem_limit\":" << int(header_attributes.shmemLimitConfig)
    << ",\"header_partitioned_cache\":" << int(header_attributes.partitionedGCConfig)
    << ",\"header_carveout\":" << state.carveoutConfig
    << ",\"header_max_dynamic\":" << header_attributes.maxDynamicSharedSizeBytes
    << ",\"driver_version\":" << driver << ",\"runtime_version\":" << toolkit
    << ",\"driver_version_error\":" << int(driver_status) << ",\"runtime_version_error\":" << int(toolkit_status)
    << ",\"flags\":" << flags << "}\n";
  return 0;
}
