# Parallel Computing Lab Practicals

Simple practical-exam programs for OpenMP, MPI, CUDA, and PyTorch.

## Folders
- OpenMP: practicals 1–5
- MPI: practicals 6–7
- CUDA: practicals 8–9
- PyTorch: practicals 10–11

## Compile and run

### OpenMP
```bash
gcc -fopenmp OpenMP/01_openmp_directives.c -o openmp
./openmp
```

### MPI
```bash
mpicc MPI/06_mpi_scatter_reduce.c -o mpi_sum
mpirun -np 4 ./mpi_sum

mpicc MPI/07_mpi_send_recv_bcast.c -o mpi_communication
mpirun -np 4 ./mpi_communication
```

### CUDA
Requires NVIDIA CUDA Toolkit / NVCC and a CUDA-capable GPU.
```bash
nvcc CUDA/08_gpu_properties.cu -o gpu_properties
./gpu_properties

nvcc CUDA/09_vector_add_compare.cu -o vector_add
./vector_add
```

### PyTorch
```bash
pip install torch
python PyTorch/10_linear_regression.py
python PyTorch/11_feedforward_neural_network.py
```

For Colab CUDA, select Runtime → Change runtime type → GPU, then compile with `!nvcc file.cu -o app` and run with `!./app`.

## Practicals
1. OpenMP directives: parallel, for, sections, single
2. OpenMP matrix multiplication and thread timing
3. Sequential and parallel merge sort with speedup
4. OpenMP threadprivate and copyin
5. OpenMP static, dynamic, and guided scheduling
6. MPI array sum using Scatter and Reduce
7. MPI Send, Recv, and Bcast
8. CUDA kernel launch configurations and GPU properties
9. CUDA vector addition and CPU/GPU timing
10. PyTorch linear regression and gradient descent
11. PyTorch feedforward neural network on random data

Execution times depend on hardware and workload. Small tasks may run slower in parallel because of startup overhead.
