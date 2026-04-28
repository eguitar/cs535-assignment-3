#!/bin/bash
#SBATCH --job-name=mm_2d
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mm_2d_%j.out

# MM-2D performance study
# P must be a perfect square: P=4 (2x2 grid), P=16 (4x4 grid)
# m, n, q must be divisible by sqrt(P):
#   P=4  -> divisible by 2
#   P=16 -> divisible by 4
# Using multiples of 4 throughout to satisfy both

module load mpi
cd ~/cs5350

echo "=== MM-2D Performance Study ==="

# ---- Vary P, square matrix ----
echo "--- Square 1024x1024x1024, vary P ---"
srun -n 4  ./MM-2D/MM-2D 1024 1024 1024
srun -n 16 ./MM-2D/MM-2D 1024 1024 1024

# ---- Vary m (tall A) ----
echo "--- Vary m, fixed n=q=1024 ---"
srun -n 4  ./MM-2D/MM-2D 512  1024 1024
srun -n 4  ./MM-2D/MM-2D 1024 1024 1024
srun -n 4  ./MM-2D/MM-2D 2048 1024 1024
srun -n 16 ./MM-2D/MM-2D 512  1024 1024
srun -n 16 ./MM-2D/MM-2D 1024 1024 1024
srun -n 16 ./MM-2D/MM-2D 2048 1024 1024

# ---- Vary q (wide C) ----
echo "--- Vary q, fixed m=n=1024 ---"
srun -n 4  ./MM-2D/MM-2D 1024 1024 512
srun -n 4  ./MM-2D/MM-2D 1024 1024 1024
srun -n 4  ./MM-2D/MM-2D 1024 1024 2048
srun -n 16 ./MM-2D/MM-2D 1024 1024 512
srun -n 16 ./MM-2D/MM-2D 1024 1024 1024
srun -n 16 ./MM-2D/MM-2D 1024 1024 2048

# ---- Vary n (inner dim) ----
echo "--- Vary n, fixed m=q=1024 ---"
srun -n 4  ./MM-2D/MM-2D 1024 512  1024
srun -n 4  ./MM-2D/MM-2D 1024 1024 1024
srun -n 4  ./MM-2D/MM-2D 1024 2048 1024
srun -n 16 ./MM-2D/MM-2D 1024 512  1024
srun -n 16 ./MM-2D/MM-2D 1024 1024 1024
srun -n 16 ./MM-2D/MM-2D 1024 2048 1024

echo "=== MM-2D done ==="
