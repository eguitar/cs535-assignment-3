// mpicc MM-2D.c -o MM-2D -lm
// mpirun -np P ./MM-2D m n q
//
// Data layout (P must be a perfect square, s = sqrt(P)):
//   Processes arranged in s×s grid; rank r -> P_{r/s, r%s}.
//   A (mxn): P_{i,j} owns block rows [i*bm..(i+1)*bm), cols [j*bn..(j+1)*bn)
//   B (nxq): P_{i,j} owns block rows [i*bn..(i+1)*bn), cols [j*bq..(j+1)*bq)
//   C (mxq): P_{i,j} computes block rows [i*bm..(i+1)*bm), cols [j*bq..(j+1)*bq)
//
// Algorithm (s steps):
//   For k = 0 .. s-1:
//     1. Row broadcast: P_{gr,k} sends its A_block to all P_{gr,j} (j != k)
//     2. Col broadcast: P_{k,gc} sends its B_block to all P_{i,gc} (i != k)
//     3. Local multiply: C_block += A_cur * B_cur
//
// Row and column broadcast phases are fully separated — no deadlock.
// Assumes m, n, q are divisible by s = sqrt(P).

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

double *alloc_mat(int r, int c) {
    return calloc((size_t)r * c, sizeof(double));
}

void randomize(double *M, int r, int c) {
    for (int i = 0; i < r * c; i++)
        M[i] = (double)rand() / RAND_MAX;
}

void serial_multiply(double *A, double *B, double *C, int m, int n, int q) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < q; k++)
                C[i*q + k] += A[i*n + j] * B[j*q + k];
}

// C (rm x cn) += A (rm x kk) * B (kk x cn), all row-major
void block_mul(double *A, double *B, double *C, int rm, int kk, int cn) {
    for (int i = 0; i < rm; i++)
        for (int k = 0; k < kk; k++)
            for (int j = 0; j < cn; j++)
                C[i*cn + j] += A[i*kk + k] * B[k*cn + j];
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

    int s = (int)round(sqrt((double)P));
    if (s * s != P) {
        if (rank == 0) printf("Error: P must be a perfect square for MM-2D.\n");
        MPI_Finalize();
        return 1;
    }

    if (m % s != 0 || n % s != 0 || q % s != 0) {
        if (rank == 0) printf("Error: m, n, q must each be divisible by sqrt(P) = %d.\n", s);
        MPI_Finalize();
        return 1;
    }

    int bm = m / s;
    int bn = n / s;
    int bq = q / s;

    int gr = rank / s;
    int gc = rank % s;

    double *A_block = alloc_mat(bm, bn);
    double *B_block = alloc_mat(bn, bq);
    double *C_block = alloc_mat(bm, bq);
    double *A_cur   = alloc_mat(bm, bn);
    double *B_cur   = alloc_mat(bn, bq);

    // A_full and B_full kept alive at rank 0 for correctness check
    double *A_full = NULL, *B_full = NULL;

    // ---- Rank 0 generates and distributes data ----
    if (rank == 0) {
        srand((unsigned)time(NULL));
        A_full = alloc_mat(m, n);
        B_full = alloc_mat(n, q);
        randomize(A_full, m, n);
        randomize(B_full, n, q);

        // Copy rank 0's own blocks (P_{0,0})
        for (int i = 0; i < bm; i++)
            for (int j = 0; j < bn; j++)
                A_block[i*bn + j] = A_full[i*n + j];
        for (int i = 0; i < bn; i++)
            for (int j = 0; j < bq; j++)
                B_block[i*bq + j] = B_full[i*q + j];

        double *A_buf = alloc_mat(bm, bn);
        double *B_buf = alloc_mat(bn, bq);

        for (int r = 1; r < P; r++) {
            int rr = r / s;
            int rc = r % s;

            for (int i = 0; i < bm; i++)
                for (int j = 0; j < bn; j++)
                    A_buf[i*bn + j] = A_full[(rr*bm + i)*n + (rc*bn + j)];
            MPI_Send(A_buf, bm * bn, MPI_DOUBLE, r, 100, MPI_COMM_WORLD);

            for (int i = 0; i < bn; i++)
                for (int j = 0; j < bq; j++)
                    B_buf[i*bq + j] = B_full[(rr*bn + i)*q + (rc*bq + j)];
            MPI_Send(B_buf, bn * bq, MPI_DOUBLE, r, 101, MPI_COMM_WORLD);
        }
        free(A_buf);
        free(B_buf);
    } else {
        MPI_Recv(A_block, bm * bn, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_block, bn * bq, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ---- 2D broadcast MM (s steps) ----
    double start = MPI_Wtime();

    for (int k = 0; k < s; k++) {
        // Phase 1: Row broadcast of A_{gr,k} from P_{gr,k} to all P_{gr,j}
        if (gc == k) {
            for (int i = 0; i < bm * bn; i++)
                A_cur[i] = A_block[i];
            for (int j = 0; j < s; j++)
                if (j != k)
                    MPI_Send(A_block, bm * bn, MPI_DOUBLE, gr*s + j, k, MPI_COMM_WORLD);
        } else {
            MPI_Recv(A_cur, bm * bn, MPI_DOUBLE, gr*s + k, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Phase 2: Column broadcast of B_{k,gc} from P_{k,gc} to all P_{i,gc}
        if (gr == k) {
            for (int i = 0; i < bn * bq; i++)
                B_cur[i] = B_block[i];
            for (int i = 0; i < s; i++)
                if (i != k)
                    MPI_Send(B_block, bn * bq, MPI_DOUBLE, i*s + gc, s + k, MPI_COMM_WORLD);
        } else {
            MPI_Recv(B_cur, bn * bq, MPI_DOUBLE, k*s + gc, s + k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Phase 3: Local block multiply
        block_mul(A_cur, B_cur, C_block, bm, bn, bq);
    }

    double end = MPI_Wtime();

    // ---- Gather C at rank 0, verify, and report ----
    if (rank == 0) {
        double *full_C = alloc_mat(m, q);

        for (int i = 0; i < bm; i++)
            for (int j = 0; j < bq; j++)
                full_C[i*q + j] = C_block[i*bq + j];

        double *C_buf = alloc_mat(bm, bq);
        for (int r = 1; r < P; r++) {
            int rr = r / s;
            int rc = r % s;
            MPI_Recv(C_buf, bm * bq, MPI_DOUBLE, r, 2*s, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < bm; i++)
                for (int j = 0; j < bq; j++)
                    full_C[(rr*bm + i)*q + (rc*bq + j)] = C_buf[i*bq + j];
        }
        free(C_buf);

        // Serial correctness check using the same A_full and B_full
        double *C_serial = alloc_mat(m, q);
        serial_multiply(A_full, B_full, C_serial, m, n, q);

        int correct = 1;
        for (int i = 0; i < m * q && correct; i++)
            if (fabs(full_C[i] - C_serial[i]) > 1e-9)
                correct = 0;

        printf("MM-2D MPI:\n\tm = %d\n\tn = %d\n\tq = %d\n\tP = %d\n\tgrid = %dx%d\n\truntime = %.9fs\n\tcorrectness = %s\n",
               m, n, q, P, s, s, end - start, correct ? "PASSED" : "FAILED");

        free(full_C);
        free(C_serial);
        free(A_full);
        free(B_full);
    } else {
        MPI_Send(C_block, bm * bq, MPI_DOUBLE, 0, 2*s, MPI_COMM_WORLD);
    }

    free(A_block); free(B_block); free(C_block); free(A_cur); free(B_cur);
    MPI_Finalize();
    return 0;
}
