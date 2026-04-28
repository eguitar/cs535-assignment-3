#!/bin/bash
#SBATCH --job-name=mm_perf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mm_perf2_%j.out

export PATH=/apps/openmpi3/bin:$PATH
export LD_LIBRARY_PATH=/apps/openmpi3/lib:$LD_LIBRARY_PATH
cd /data03/home/edtrinh/cs5350

echo "=============================="
echo "=== SERIAL BASELINE (P=1) ==="
echo "=============================="
for size in 512 1024 2048; do
    mpirun -np 1 ./MM-ser/MM-ser $size $size $size
done

echo ""
echo "=== MM-1D: Vary P, square ==="
for P in 1 2 4 8 16; do
    mpirun -np $P ./MM-1D/MM-1D 1024 1024 1024
done

echo ""
echo "=== MM-1D: Vary m (P=16) ==="
for m in 512 1024 2048; do
    mpirun -np 16 ./MM-1D/MM-1D $m 1024 1024
done

echo ""
echo "=== MM-1D: Vary q (P=16) ==="
for q in 512 1024 2048; do
    mpirun -np 16 ./MM-1D/MM-1D 1024 1024 $q
done

echo ""
echo "=== MM-1D: Vary n (P=16) ==="
for n in 512 1024 2048; do
    mpirun -np 16 ./MM-1D/MM-1D 1024 $n 1024
done

echo ""
echo "=== MM-2D: Vary P, square ==="
for P in 4 16; do
    mpirun -np $P ./MM-2D/MM-2D 1024 1024 1024
done

echo ""
echo "=== MM-2D: Vary m (P=16) ==="
for m in 512 1024 2048; do
    mpirun -np 16 ./MM-2D/MM-2D $m 1024 1024
done

echo ""
echo "=== MM-2D: Vary q (P=16) ==="
for q in 512 1024 2048; do
    mpirun -np 16 ./MM-2D/MM-2D 1024 1024 $q
done

echo ""
echo "=== MM-2D: Vary n (P=16) ==="
for n in 512 1024 2048; do
    mpirun -np 16 ./MM-2D/MM-2D 1024 $n 1024
done

echo "=== ALL DONE ==="