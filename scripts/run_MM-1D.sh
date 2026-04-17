#!/bin/bash
#SBATCH --job-name=mm_1d
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mm_1d_%j.out

# MM-1D performance study
# Varies P (1,2,4,8,16) and matrix shape
# m and q must be divisible by P — using multiples of 16 throughout

module load mpi
cd ~/cs5350

echo "=== MM-1D Performance Study ==="

# ---- Vary P, square matrix ----
echo "--- Square 1024x1024x1024, vary P ---"
srun -n 1  ./MM-1D/MM-1D 1024 1024 1024
srun -n 2  ./MM-1D/MM-1D 1024 1024 1024
srun -n 4  ./MM-1D/MM-1D 1024 1024 1024
srun -n 8  ./MM-1D/MM-1D 1024 1024 1024
srun -n 16 ./MM-1D/MM-1D 1024 1024 1024

# ---- Vary m (tall A), P=16 ----
echo "--- Vary m, fixed n=q=1024, P=16 ---"
srun -n 16 ./MM-1D/MM-1D 512  1024 1024
srun -n 16 ./MM-1D/MM-1D 1024 1024 1024
srun -n 16 ./MM-1D/MM-1D 2048 1024 1024

# ---- Vary q (wide C), P=16 ----
echo "--- Vary q, fixed m=n=1024, P=16 ---"
srun -n 16 ./MM-1D/MM-1D 1024 1024 512
srun -n 16 ./MM-1D/MM-1D 1024 1024 1024
srun -n 16 ./MM-1D/MM-1D 1024 1024 2048

# ---- Vary n (inner dim), P=16 ----
echo "--- Vary n, fixed m=q=1024, P=16 ---"
srun -n 16 ./MM-1D/MM-1D 1024 512  1024
srun -n 16 ./MM-1D/MM-1D 1024 1024 1024
srun -n 16 ./MM-1D/MM-1D 1024 2048 1024

echo "=== MM-1D done ==="
