// mpicc MM-ser.c -o MM-ser
// mpirun -np 1 ./MM-ser m n q

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

double *allocate_matrix(int r, int c) {
    return calloc((size_t)r * c, sizeof(double));
}

void randomize_matrix(double *M, int r, int c) {
    for (int i = 0; i < r * c; i++)
        M[i] = (double)rand() / RAND_MAX;
}

void matrix_multiply(double *A, double *B, double *C, int m, int n, int q) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < q; k++)
                C[i*q + k] += A[i*n + j] * B[j*q + k];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    if (argc != 4) {
        printf("Usage: %s m n q\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    srand((unsigned)time(NULL));

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int q = atoi(argv[3]);

    double *A = allocate_matrix(m, n);
    double *B = allocate_matrix(n, q);
    double *C = allocate_matrix(m, q);

    randomize_matrix(A, m, n);
    randomize_matrix(B, n, q);

    double start = MPI_Wtime();
    matrix_multiply(A, B, C, m, n, q);
    double end = MPI_Wtime();

    printf("Serial MM (MPI):\n\tm = %d\n\tn = %d\n\tq = %d\n\truntime = %.9fs\n",
           m, n, q, end - start);

    free(A);
    free(B);
    free(C);
    MPI_Finalize();
    return 0;
}
