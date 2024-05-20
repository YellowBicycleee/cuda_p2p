#include <cstdio>
#include <mpi.h>
#include "multi_gpu.h"


int main(int argc, char *argv[]) { 
    MPI_Init(&argc, &argv);
    
    multi_gpu_test();

    MPI_Finalize();
    return 0;
}