#!/bin/bash
#SBATCH --job-name=mm_serial
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=mm_ser_%j.out

module load mpi
cd ~/cs5350

echo "=== MM-ser (Serial Baseline) ==="

# Square matrices
echo "--- Square ---"
srun -n 1 ./MM-ser/MM-ser 1024 1024 1024
srun -n 1 ./MM-ser/MM-ser 512  512  512

# Vary m (tall A)
echo "--- Vary m (tall A, fixed n=q=1024) ---"
srun -n 1 ./MM-ser/MM-ser 512  1024 1024
srun -n 1 ./MM-ser/MM-ser 1024 1024 1024
srun -n 1 ./MM-ser/MM-ser 2048 1024 1024

# Vary q (wide C)
echo "--- Vary q (wide C, fixed m=n=1024) ---"
srun -n 1 ./MM-ser/MM-ser 1024 1024 512
srun -n 1 ./MM-ser/MM-ser 1024 1024 1024
srun -n 1 ./MM-ser/MM-ser 1024 1024 2048

# Vary n (inner dimension)
echo "--- Vary n (inner dim, fixed m=q=1024) ---"
srun -n 1 ./MM-ser/MM-ser 1024 512  1024
srun -n 1 ./MM-ser/MM-ser 1024 1024 1024
srun -n 1 ./MM-ser/MM-ser 1024 2048 1024

echo "=== MM-ser done ==="
