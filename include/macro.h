#pragma once

#define CHECK_MPI(cmd)                         \
  do {                                         \
    int err = cmd;                             \
    if (err != MPI_SUCCESS) {                  \
      fprintf(stderr, "MPI error: %d\n", err); \
      exit(1);                                 \
    }                                          \
  } while (0)

#define CHECK_CUDA(cmd)                                                                                   \
  do {                                                                                                    \
    cudaError_t err = cmd;                                                                                \
    if (err != cudaSuccess) {                                                                             \
      fprintf(stderr, "CUDA error: %s, file %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(1);                                                                                            \
    }                                                                                                     \
  } while (0)
