// mpicc MM-1D.c -o MM-1D
// mpirun -np P ./MM-1D m n q
//
// Data layout:
//   A (mxn): row-striped — Pi owns rows [i*m/P .. (i+1)*m/P)
//   B (nxq): column-striped — Pi owns cols [i*q/P .. (i+1)*q/P)
//   C (mxq): row-striped — Pi owns rows [i*m/P .. (i+1)*m/P)
//
// Algorithm: ring rotation of B column blocks.
// Each of P steps: compute partial C using current B block, then
// pass B block to (rank+1)%P and receive from (rank-1+P)%P.
// Even/odd ordering avoids deadlock with blocking MPI_Send/Recv.
//
// Assumes m and q are divisible by P.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

double *allocate_matrix(int r, int c) {
    return calloc((size_t)r * c, sizeof(double));
}

void randomize_matrix(double *M, int r, int c) {
    for (int i = 0; i < r * c; i++)
        M[i] = (double)rand() / RAND_MAX;
}

void matrix_multiply_serial(double *A, double *B, double *C, int m, int n, int q) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < q; k++)
                C[i*q + k] += A[i*n + j] * B[j*q + k];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc != 4) {
        if (rank == 0) printf("Usage: %s m n q\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int q = atoi(argv[3]);

    if (m % P != 0 || q % P != 0) {
        if (rank == 0) printf("Error: m and q must be divisible by P.\n");
        MPI_Finalize();
        return 1;
    }

    int rpp = m / P;   // rows per process
    int cpp = q / P;   // cols per process (from B)

    double *local_A = allocate_matrix(rpp, n);
    double *local_B = allocate_matrix(n, cpp);
    double *local_C = allocate_matrix(rpp, q);
    double *cur_B   = allocate_matrix(n, cpp);
    double *recv_B  = allocate_matrix(n, cpp);

    // A_full and B_full kept alive at rank 0 for correctness check after gather
    double *A_full = NULL, *B_full = NULL;

    // ---- Distribute data from rank 0 ----
    if (rank == 0) {
        srand((unsigned)time(NULL));
        A_full = allocate_matrix(m, n);
        B_full = allocate_matrix(n, q);
        randomize_matrix(A_full, m, n);
        randomize_matrix(B_full, n, q);

        // Copy rank 0's own rows of A
        for (int i = 0; i < rpp * n; i++)
            local_A[i] = A_full[i];

        // Copy rank 0's own cols of B (cols 0..cpp-1)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < cpp; j++)
                local_B[k*cpp + j] = B_full[k*q + j];

        // Send to all other ranks
        double *B_buf = allocate_matrix(n, cpp);
        for (int r = 1; r < P; r++) {
            MPI_Send(&A_full[r * rpp * n], rpp * n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            for (int k = 0; k < n; k++)
                for (int j = 0; j < cpp; j++)
                    B_buf[k*cpp + j] = B_full[k*q + r*cpp + j];
            MPI_Send(B_buf, n * cpp, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
        }
        free(B_buf);
    } else {
        MPI_Recv(local_A, rpp * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_B, n * cpp, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ---- Ring rotation ----
    for (int i = 0; i < n * cpp; i++)
        cur_B[i] = local_B[i];

    int cur_owner = rank;

    double start = MPI_Wtime();

    for (int step = 0; step < P; step++) {
        int col_off = cur_owner * cpp;
        for (int i = 0; i < rpp; i++)
            for (int j = 0; j < cpp; j++)
                for (int k = 0; k < n; k++)
                    local_C[i*q + col_off + j] += local_A[i*n + k] * cur_B[k*cpp + j];

        if (step < P - 1) {
            int send_to   = (rank + 1) % P;
            int recv_from = (rank - 1 + P) % P;

            if (rank % 2 == 0) {
                MPI_Send(cur_B,  n * cpp, MPI_DOUBLE, send_to,   2, MPI_COMM_WORLD);
                MPI_Recv(recv_B, n * cpp, MPI_DOUBLE, recv_from, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(recv_B, n * cpp, MPI_DOUBLE, recv_from, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(cur_B,  n * cpp, MPI_DOUBLE, send_to,   2, MPI_COMM_WORLD);
            }

            double *tmp = cur_B; cur_B = recv_B; recv_B = tmp;
            cur_owner = (cur_owner - 1 + P) % P;
        }
    }

    double end = MPI_Wtime();

    // ---- Gather C at rank 0, verify, and report ----
    if (rank == 0) {
        double *full_C = allocate_matrix(m, q);
        for (int i = 0; i < rpp * q; i++)
            full_C[i] = local_C[i];
        for (int r = 1; r < P; r++)
            MPI_Recv(&full_C[r * rpp * q], rpp * q, MPI_DOUBLE, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Serial correctness check using the same A_full and B_full
        double *C_serial = allocate_matrix(m, q);
        matrix_multiply_serial(A_full, B_full, C_serial, m, n, q);

        int correct = 1;
        for (int i = 0; i < m * q && correct; i++)
            if (fabs(full_C[i] - C_serial[i]) > 1e-9)
                correct = 0;

        printf("MM-1D MPI:\n\tm = %d\n\tn = %d\n\tq = %d\n\tP = %d\n\truntime = %.9fs\n\tcorrectness = %s\n",
               m, n, q, P, end - start, correct ? "PASSED" : "FAILED");

        free(full_C);
        free(C_serial);
        free(A_full);
        free(B_full);
    } else {
        MPI_Send(local_C, rpp * q, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }

    free(local_A); free(local_B); free(local_C); free(cur_B); free(recv_B);
    MPI_Finalize();
    return 0;
}
