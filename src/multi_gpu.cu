#include <mpi.h>

#include "macro.h"
#include "multi_gpu.h"

#define MAX_PROCESS_NUM 16

#define print(condition, format, ...) \
  do                                  \
    if (condition) {                  \
      printf(format, ##__VA_ARGS__);  \
    }                                 \
  while (0)

static int mpi_rank;
static int mpi_size;

static int gpu_num;
static int my_gpu_id;

constexpr int ARRAY_SIZE = 1024 * 1024;
static int *local_memory;
static int *neighbor_memory;
static int *redundant_memory;

__global__ void init_arry(int *arr, int proc_rank, int length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < length; i += stride) {
    arr[i] = i * 100 + proc_rank;
  }
}

__global__ void print_array(int *arr, int proc_rank, int length) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // int stride = blockDim.x * gridDim.x;
  //    for (int i = idx; i < length; i += stride) {
  //      printf("%d ", arr[i]);
  //    }
  if (length == 0) {
    printf(
        "in function print_arr, my mpi rank is %d, first four elems are \
        %d %d %d %d\nlast four elems are %d %d %d %d\n",
        proc_rank, arr[0], arr[1], arr[2], arr[3], arr[ARRAY_SIZE - 4], arr[ARRAY_SIZE - 3], arr[ARRAY_SIZE - 2],
        arr[ARRAY_SIZE - 1]);
  }
}

void multi_gpu_test() {
  int src_rank;
  int dst_rank;

  MPI_Request send_req;
  MPI_Request recv_req;

  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
  CHECK_CUDA(cudaGetDeviceCount(&gpu_num));
  src_rank = (mpi_rank + mpi_size - 1) % mpi_size;
  dst_rank = (mpi_rank + 1) % mpi_size;
  my_gpu_id = mpi_rank % gpu_num;
  CHECK_CUDA(cudaSetDevice(my_gpu_id));

  if (mpi_size == gpu_num) {
    print(mpi_rank == 0, "mpi_size and gpu_num is the same %d\n", mpi_size);
  } else {
    printf("mpi_size = %d, mpi_rank = %d, gpu_id = %d\n", mpi_size, mpi_rank, my_gpu_id);
  }

  // init local memory
  CHECK_CUDA(cudaMalloc(&local_memory, ARRAY_SIZE * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&redundant_memory, ARRAY_SIZE * sizeof(int)));
  init_arry<<<ARRAY_SIZE / 256, 256>>>(local_memory, mpi_rank, ARRAY_SIZE);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  print(mpi_rank == 0, "data init over\n");

  // comm
  // 创建显存句柄，并发送或接收
  cudaIpcMemHandle_t handle;
  cudaIpcMemHandle_t pre_handle;
  cudaIpcGetMemHandle(&handle, local_memory);

  // int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
  // int tag, MPI_Comm comm, MPI_Request *request)
  MPI_Isend(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD, &send_req);
  // int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int
  // tag, MPI_Comm comm, MPI_Request *request)
  MPI_Irecv(&pre_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, &recv_req);
  MPI_Wait(&send_req, MPI_STATUS_IGNORE);
  MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
  CHECK_CUDA(cudaIpcOpenMemHandle((void **)&neighbor_memory, pre_handle, cudaIpcMemLazyEnablePeerAccess));

  CHECK_CUDA(cudaMemcpy(redundant_memory, neighbor_memory, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
  // print
  print_array<<<1, 1>>>(redundant_memory, mpi_rank, 0);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(local_memory));
  CHECK_CUDA(cudaFree(redundant_memory));
  CHECK_CUDA(cudaIpcCloseMemHandle(neighbor_memory));
}