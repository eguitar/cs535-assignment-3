#!/bin/bash
#SBATCH --job-name=mm_correctness
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=correctness_%j.out

# Small matrix correctness test for MM-1D and MM-2D
# Run BEFORE any large performance experiments

module load mpi
cd ~/cs5350

echo "=== Compiling ==="
mpicc MM-ser/MM-ser.c -o MM-ser/MM-ser
mpicc MM-1D/MM-1D.c  -o MM-1D/MM-1D  -lm
mpicc MM-2D/MM-2D.c  -o MM-2D/MM-2D  -lm
echo "Compile done"

echo ""
echo "=== MM-1D Correctness (32x32x32) ==="
srun -n 4  ./MM-1D/MM-1D  32  32  32
srun -n 8  ./MM-1D/MM-1D  32  32  32
srun -n 16 ./MM-1D/MM-1D  32  32  32

echo ""
echo "=== MM-2D Correctness (16x16x16) ==="
srun -n 4  ./MM-2D/MM-2D  16  16  16
srun -n 16 ./MM-2D/MM-2D  16  16  16

echo ""
echo "=== Correctness tests done ==="
