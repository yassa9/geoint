#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define P0_DIAM_MIN_M 100.0
#define P0_DIAM_MAX_M 1000.0
#define SEP_MIN 5.0
#define SEP_MAX 30.0
#define MIN_SIDE_M 100.0
#define MAX_SIDE_M 20000.0
#define M_LAT 111320.0
#define PI 3.14159265358979323846

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void match_kernel(
    const double *lon, const double *lat, const double *area, const double *diam,
    const long long *triples, long long n_triples,
    double ang_lo, double ang_hi, double ratio_lo, double ratio_hi,
    long long *out_p0, long long *out_p1, long long *out_p2,
    double *out_angle, double *out_ratio, unsigned long long *out_count)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n_triples) return;

    long long idx[3] = {triples[3 * i], triples[3 * i + 1], triples[3 * i + 2]};
    double a[3] = {area[idx[0]], area[idx[1]], area[idx[2]]};

    int pos[3] = {0, 1, 2};
    for (int a1 = 1; a1 < 3; a1++) {
        int key = pos[a1];
        double keyval = a[key];
        int j = a1 - 1;
        while (j >= 0 && a[pos[j]] > keyval) {
            pos[j + 1] = pos[j];
            j--;
        }
        pos[j + 1] = key;
    }

    long long p0idx = idx[pos[0]];
    long long aidx = idx[pos[1]];
    long long bidx = idx[pos[2]];

    double lon0 = lon[p0idx], lat0 = lat[p0idx], diam0 = diam[p0idx];
    double lona = lon[aidx], lata = lat[aidx];
    double lonb = lon[bidx], latb = lat[bidx];

    double m_lon = M_LAT * cos(lat0 * PI / 180.0);
    double xa = (lona - lon0) * m_lon, ya = (lata - lat0) * M_LAT;
    double xb = (lonb - lon0) * m_lon, yb = (latb - lat0) * M_LAT;

    double cross = xa * yb - xb * ya;
    bool ccw = cross > 0;

    double x1 = ccw ? xa : xb, y1 = ccw ? ya : yb;
    double x2 = ccw ? xb : xa, y2 = ccw ? yb : ya;
    long long p1idx = ccw ? aidx : bidx;
    long long p2idx = ccw ? bidx : aidx;

    double d01 = hypot(x1, y1);
    double d02 = hypot(x2, y2);
    double cos_ang = (x1 * x2 + y1 * y2) / (d01 * d02 + 1e-9);
    cos_ang = fmin(1.0, fmax(-1.0, cos_ang));
    double angle0 = acos(cos_ang) * 180.0 / PI;
    double ratio = d01 / (d02 + 1e-9);
    double sep = d01 / (diam0 + 1e-9);

    bool hit = (cross != 0) &&
               (diam0 >= P0_DIAM_MIN_M) && (diam0 <= P0_DIAM_MAX_M) &&
               (sep >= SEP_MIN) && (sep <= SEP_MAX) &&
               (angle0 >= ang_lo) && (angle0 <= ang_hi) &&
               (ratio >= ratio_lo) && (ratio <= ratio_hi) &&
               (d01 >= MIN_SIDE_M) && (d01 <= MAX_SIDE_M) &&
               (d02 >= MIN_SIDE_M) && (d02 <= MAX_SIDE_M);

    if (hit) {
        unsigned long long slot = atomicAdd(out_count, 1ULL);
        out_p0[slot] = p0idx;
        out_p1[slot] = p1idx;
        out_p2[slot] = p2idx;
        out_angle[slot] = angle0;
        out_ratio[slot] = ratio;
    }
}

static void *read_block(FILE *f, size_t n) {
    void *buf = malloc(n);
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "short read\n");
        exit(1);
    }
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("gpu: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("vram: %.0f/%.0f MB free\n", free_mem / 1e6, total_mem / 1e6);

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    long long n_points, n_triples;
    double ang_lo, ang_hi, ratio_lo, ratio_hi;
    fread(&n_points, sizeof(long long), 1, fin);
    fread(&n_triples, sizeof(long long), 1, fin);
    fread(&ang_lo, sizeof(double), 1, fin);
    fread(&ang_hi, sizeof(double), 1, fin);
    fread(&ratio_lo, sizeof(double), 1, fin);
    fread(&ratio_hi, sizeof(double), 1, fin);

    double *lon = (double *)read_block(fin, n_points * sizeof(double));
    double *lat = (double *)read_block(fin, n_points * sizeof(double));
    double *area = (double *)read_block(fin, n_points * sizeof(double));
    double *diam = (double *)read_block(fin, n_points * sizeof(double));
    long long *triples = (long long *)read_block(fin, n_triples * 3 * sizeof(long long));
    fclose(fin);

    printf("points: %lld, triples: %lld\n", n_points, n_triples);

    double *d_lon, *d_lat, *d_area, *d_diam;
    long long *d_triples, *d_out_p0, *d_out_p1, *d_out_p2;
    double *d_out_angle, *d_out_ratio;
    unsigned long long *d_out_count;

    CUDA_CHECK(cudaMalloc(&d_lon, n_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lat, n_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_area, n_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diam, n_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_triples, n_triples * 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_out_p0, n_triples * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_out_p1, n_triples * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_out_p2, n_triples * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_out_angle, n_triples * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out_ratio, n_triples * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_lon, lon, n_points * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lat, lat, n_points * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area, area, n_points * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diam, diam, n_points * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_triples, triples, n_triples * 3 * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(unsigned long long)));

    size_t free_after, total_after;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
    printf("vram used: %.0f MB\n", (free_mem - free_after) / 1e6);

    int threads = 256;
    int blocks = (int)((n_triples + threads - 1) / threads);

    auto t0 = std::chrono::steady_clock::now();
    match_kernel<<<blocks, threads>>>(d_lon, d_lat, d_area, d_diam, d_triples, n_triples,
                                       ang_lo, ang_hi, ratio_lo, ratio_hi,
                                       d_out_p0, d_out_p1, d_out_p2, d_out_angle, d_out_ratio, d_out_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("kernel time: %.1f ms\n", ms);

    unsigned long long n_hits;
    CUDA_CHECK(cudaMemcpy(&n_hits, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    printf("hits: %llu\n", n_hits);

    long long *p0 = (long long *)malloc(n_hits * sizeof(long long));
    long long *p1 = (long long *)malloc(n_hits * sizeof(long long));
    long long *p2 = (long long *)malloc(n_hits * sizeof(long long));
    double *angle = (double *)malloc(n_hits * sizeof(double));
    double *ratio = (double *)malloc(n_hits * sizeof(double));

    CUDA_CHECK(cudaMemcpy(p0, d_out_p0, n_hits * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p1, d_out_p1, n_hits * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p2, d_out_p2, n_hits * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(angle, d_out_angle, n_hits * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ratio, d_out_ratio, n_hits * sizeof(double), cudaMemcpyDeviceToHost));

    FILE *fout = fopen(argv[2], "wb");
    long long n_hits_ll = (long long)n_hits;
    fwrite(&n_hits_ll, sizeof(long long), 1, fout);
    fwrite(p0, sizeof(long long), n_hits, fout);
    fwrite(p1, sizeof(long long), n_hits, fout);
    fwrite(p2, sizeof(long long), n_hits, fout);
    fwrite(angle, sizeof(double), n_hits, fout);
    fwrite(ratio, sizeof(double), n_hits, fout);
    fclose(fout);

    return 0;
}
