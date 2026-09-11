/*
 * amx_tfs_v3k.cpp -- TFS Fusion V3 with variable K
 *
 * Supports K = 32, 64, 128, 256 (any multiple of 32)
 * Same algorithm as V3: degree-sorted + prefetch + smart memset
 *
 * Compile:
 *   icpx -O3 -march=sapphirerapids -mamx-bf16 -mamx-tile \
 *        -mavx512bf16 -qopenmp -o amx_tfs_v3k amx_tfs_v3k.cpp
 */

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <omp.h>

#define TR   16
#define TC0  0
#define TC1  1
#define TC2  2
#define TC3  3
#define TA   4
#define TB0  5
#define TB1  6

struct __attribute__((aligned(64))) tilecfg_t {
    uint8_t  palette;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

static void setup_tilecfg(tilecfg_t *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->palette = 1;
    for (int t = 0; t < 7; t++) {
        cfg->rows[t]  = TR;
        cfg->colsb[t] = 64;
    }
}

static inline uint16_t f32_to_bf16(float f)
{
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static inline float bf16_to_f32(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

/* ================================================================
 * VNNI packing for W[K_in × K_out]
 * Wv layout: [KB][NB][16][32] flat uint16_t
 * KB = K/32, NB = K/16
 * ================================================================ */
static void make_W_vnni(const float *W, uint16_t *Wv, int K)
{
    int KB = K / 32;
    int NB = K / 16;
    for (int kb = 0; kb < KB; kb++)
        for (int ob = 0; ob < NB; ob++)
            for (int kp = 0; kp < 16; kp++)
                for (int n = 0; n < 16; n++) {
                    int k0 = kb*32 + kp*2, k1 = k0 + 1;
                    int col = ob*16 + n;
                    uint16_t v0 = f32_to_bf16(W[k0*K + col]);
                    uint16_t v1 = f32_to_bf16(W[k1*K + col]);
                    int base = ((kb*NB + ob)*16 + kp)*32 + n*2;
                    memcpy(&Wv[base], &v0, 2);
                    memcpy(&Wv[base+1], &v1, 2);
                }
}

static void convert_H_bf16(const float *H, uint16_t *Hb, int N, int K)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int k = 0; k < K; k++)
            Hb[(size_t)i*K + k] = f32_to_bf16(H[(size_t)i*K + k]);
}

/* ================================================================
 * Degree-sorted permutation (ascending)
 * ================================================================ */
static int* make_degree_perm(const uint32_t *indptr, int N)
{
    int *perm = (int*)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) perm[i] = i;
    const uint32_t *ip = indptr;
    std::sort(perm, perm + N, [ip](int a, int b) {
        return (ip[a+1] - ip[a]) < (ip[b+1] - ip[b]);
    });
    return perm;
}

static void print_deg_stats(const uint32_t *indptr, int N)
{
    uint32_t mn = ~0u, mx = 0; double sum = 0;
    for (int i = 0; i < N; i++) {
        uint32_t d = indptr[i+1] - indptr[i];
        if (d < mn) mn = d; if (d > mx) mx = d; sum += d;
    }
    printf("  deg: min=%u max=%u avg=%.1f ratio=%.0f\n",
           mn, mx, sum/N, (double)mx/(sum/N));
}

/* ================================================================
 * TFS V3 Kernel — variable K
 *
 * KB = K/32 (feature blocks)
 * NB = K/16 (output blocks)
 * NP = ceil(NB/4) (obp passes, each handles up to 4 output blocks)
 *
 * Tile allocation (7/8 used):
 *   TC0-TC3: up to 4 output accumulators (16×16 FP32 each)
 *   TA:      input features (16×32 BF16)
 *   TB0-TB1: weight slices (16×16 VNNI BF16)
 * ================================================================ */
static void tfs_v3k(
    const uint32_t *indptr, const uint32_t *indices,
    const uint16_t *Hb, const uint16_t *Wv,
    float *C, const int *perm,
    int N, int R, int K)
{
    const int KB = K / 32;
    const int NB = K / 16;
    const int NP = (NB + 3) / 4;

    memset(C, 0, (size_t)N * K * sizeof(float));

    #pragma omp parallel
    {
        if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) {
            perror("arch_prctl"); exit(1);
        }
        tilecfg_t cfg;
        setup_tilecfg(&cfg);
        _tile_loadconfig(&cfg);

        /* Thread-local buffers */
        uint16_t *Hbuf = (uint16_t*)aligned_alloc(64, TR * K * sizeof(uint16_t));
        float    Ctmp[TR * 16] __attribute__((aligned(64)));
        uint32_t base_local[TR];
        uint32_t deg_local[TR];
        int      orig_row[TR];

        #pragma omp for schedule(dynamic, 1) nowait
        for (int rg = 0; rg < N; rg += R) {
            int rg_end = (rg + R < N) ? (rg + R) : N;

            for (int i = rg; i < rg_end; i += TR) {
                int batch = ((i + TR) <= rg_end) ? TR : (rg_end - i);

                int max_deg = 0;
                for (int n = 0; n < batch; n++) {
                    int row = perm[i + n];
                    orig_row[n]   = row;
                    base_local[n] = indptr[row];
                    deg_local[n]  = indptr[row+1] - indptr[row];
                    if ((int)deg_local[n] > max_deg)
                        max_deg = (int)deg_local[n];
                }

                for (int obp = 0; obp < NP; obp++) {
                    int obs_this = NB - obp * 4;
                    if (obs_this > 4) obs_this = 4;

                    _tile_zero(TC0);
                    if (obs_this > 1) _tile_zero(TC1);
                    if (obs_this > 2) _tile_zero(TC2);
                    if (obs_this > 3) _tile_zero(TC3);

                    memset(Hbuf, 0, TR * K * sizeof(uint16_t));
                    int active_from = 0;

                    for (int s = 0; s < max_deg; s++) {

                        /* Prefetch next step */
                        if (s + 1 < max_deg) {
                            for (int n = active_from; n < batch; n++) {
                                if ((uint32_t)(s+1) < deg_local[n]) {
                                    uint32_t j_next = indices[base_local[n]+s+1];
                                    const char *addr =
                                        (const char*)&Hb[(size_t)j_next*K];
                                    for (int pf = 0; pf < K*2; pf += 64)
                                        _mm_prefetch(addr + pf, _MM_HINT_T0);
                                }
                            }
                        }

                        /* Deactivate exhausted rows */
                        while (active_from < batch &&
                               (uint32_t)s >= deg_local[active_from]) {
                            memset(&Hbuf[active_from * K], 0,
                                   K * sizeof(uint16_t));
                            active_from++;
                        }
                        if (active_from >= batch) break;

                        /* Gather active rows (full K dims) */
                        for (int n = active_from; n < batch; n++) {
                            uint32_t j = indices[base_local[n] + s];
                            memcpy(&Hbuf[n * K],
                                   &Hb[(size_t)j * K],
                                   K * sizeof(uint16_t));
                        }

                        /* KB-inner loop */
                        for (int kb = 0; kb < KB; kb++) {
                            _tile_loadd(TA,
                                (const uint8_t*)Hbuf + kb * 64,
                                K * 2);  /* stride = K * sizeof(bf16) */

                            int ob0 = obp * 4;

                            /* Wv offset: ((kb*NB + ob)*16 + kp)*32
                             * For tile_loadd with stride=64:
                             * &Wv[(kb*NB + ob) * 16 * 32] */
                            #define WV_ADDR(kb_, ob_) \
                                (&Wv[((kb_)*NB + (ob_)) * 16 * 32])

                            /* Always do first pair: C0 */
                            _tile_loadd(TB0, WV_ADDR(kb, ob0+0), 64);
                            _tile_dpbf16ps(TC0, TA, TB0);

                            if (obs_this > 1) {
                                _tile_loadd(TB1, WV_ADDR(kb, ob0+1), 64);
                                _tile_dpbf16ps(TC1, TA, TB1);
                            }
                            if (obs_this > 2) {
                                _tile_loadd(TB0, WV_ADDR(kb, ob0+2), 64);
                                _tile_dpbf16ps(TC2, TA, TB0);
                            }
                            if (obs_this > 3) {
                                _tile_loadd(TB1, WV_ADDR(kb, ob0+3), 64);
                                _tile_dpbf16ps(TC3, TA, TB1);
                            }

                            #undef WV_ADDR
                        }
                    } /* s */

                    /* Store C tiles to original positions */
                    int col = obp * 64;

                    _tile_stored(TC0, Ctmp, 64);
                    for (int n = 0; n < batch; n++)
                        memcpy(&C[(size_t)orig_row[n]*K + col + 0],
                               &Ctmp[n*16], 64);

                    if (obs_this > 1) {
                        _tile_stored(TC1, Ctmp, 64);
                        for (int n = 0; n < batch; n++)
                            memcpy(&C[(size_t)orig_row[n]*K + col + 16],
                                   &Ctmp[n*16], 64);
                    }
                    if (obs_this > 2) {
                        _tile_stored(TC2, Ctmp, 64);
                        for (int n = 0; n < batch; n++)
                            memcpy(&C[(size_t)orig_row[n]*K + col + 32],
                                   &Ctmp[n*16], 64);
                    }
                    if (obs_this > 3) {
                        _tile_stored(TC3, Ctmp, 64);
                        for (int n = 0; n < batch; n++)
                            memcpy(&C[(size_t)orig_row[n]*K + col + 48],
                                   &Ctmp[n*16], 64);
                    }
                } /* obp */
            }
        }
        free(Hbuf);
        _tile_release();
    }
}

/* ================================================================
 * Reference two-step
 * ================================================================ */
static void ref_twostep(
    const uint32_t *indptr, const uint32_t *indices,
    const uint16_t *Hb, const float *W,
    float *C_ref, int N, int K)
{
    float *Z = (float*)calloc((size_t)N * K, sizeof(float));
    if (!Z) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < N; i++)
        for (uint32_t p = indptr[i]; p < indptr[i+1]; p++) {
            uint32_t j = indices[p];
            for (int k = 0; k < K; k++)
                Z[(size_t)i*K+k] += bf16_to_f32(Hb[(size_t)j*K+k]);
        }
    for (int i = 0; i < N; i++)
        for (int n = 0; n < K; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += Z[(size_t)i*K+k] * W[k*K+n];
            C_ref[(size_t)i*K+n] = sum;
        }
    free(Z);
}

/* ================================================================
 * CSR loading
 * ================================================================ */
struct csr_t { uint32_t *indptr, *indices; int N; uint32_t nnz; };

static csr_t load_csrbin(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    uint8_t hdr[36]; fread(hdr, 1, 36, f);
    uint64_t nr, nc, nz;
    memcpy(&nr, &hdr[12], 8);
    memcpy(&nc, &hdr[20], 8);
    memcpy(&nz, &hdr[28], 8);
    int N = (int)nr; uint32_t nnz = (uint32_t)nz;
    printf("  csrbin: N=%d  NNZ=%u\n", N, nnz);
    uint32_t *indptr = (uint32_t*)malloc((size_t)(N+1)*4);
    fread(indptr, 4, N+1, f);
    uint32_t *indices = (uint32_t*)malloc((size_t)nnz*4);
    fread(indices, 4, nnz, f);
    fclose(f);
    return {indptr, indices, N, nnz};
}

/* ================================================================
 * Self-test
 * ================================================================ */
static void selftest(int K)
{
    printf("=== Self-test (N=256, avg_deg=8, K=%d) ===\n", K);
    const int N = 256;
    int KB = K/32, NB = K/16;

    srand(42);
    uint32_t *indptr = (uint32_t*)malloc((N+1)*4);
    indptr[0] = 0;
    for (int i = 0; i < N; i++) indptr[i+1] = indptr[i] + 4 + rand()%9;
    uint32_t nnz = indptr[N];
    uint32_t *indices = (uint32_t*)malloc(nnz*4);
    for (uint32_t p = 0; p < nnz; p++) indices[p] = rand()%N;

    float *H = (float*)malloc((size_t)N*K*4);
    float *W = (float*)malloc((size_t)K*K*4);
    for (int i = 0; i < N*K; i++) H[i] = 0.01f*((rand()%200)-100);
    for (int i = 0; i < K*K; i++) W[i] = 0.01f*((rand()%200)-100);

    uint16_t *Hb = (uint16_t*)aligned_alloc(64, (size_t)N*K*2);
    uint16_t *Wv = (uint16_t*)aligned_alloc(64, (size_t)KB*NB*16*32*2);
    convert_H_bf16(H, Hb, N, K);
    make_W_vnni(W, Wv, K);

    int *perm = make_degree_perm(indptr, N);

    float *C = (float*)calloc((size_t)N*K, 4);
    tfs_v3k(indptr, indices, Hb, Wv, C, perm, N, 64, K);

    float *Cr = (float*)calloc((size_t)N*K, 4);
    ref_twostep(indptr, indices, Hb, W, Cr, N, K);

    double md=0, mr=0;
    for (size_t i = 0; i < (size_t)N*K; i++) {
        double d = fabs((double)C[i]-(double)Cr[i]);
        double r = fabs((double)Cr[i]);
        if (d>md) md=d; if (r>mr) mr=r;
    }
    double rel = (mr>0) ? md/mr : 0;
    printf("  rel_err=%.6f %s\n", rel, (rel<0.02)?"PASS":"FAIL");

    free(indptr);free(indices);free(H);free(W);
    free(Hb);free(Wv);free(C);free(Cr);free(perm);
}

/* ================================================================
 * Benchmark
 * ================================================================ */
static void benchmark(const char *dir, const char *name, int R, int K)
{
    int KB = K/32, NB = K/16;
    char path[512];
    snprintf(path, 512, "%s/%s/%s.csrbin", dir, name, name);
    printf("Loading: %s\n", path);
    csr_t csr = load_csrbin(path);
    int N = csr.N; uint32_t nnz = csr.nnz;
    printf("Matrix: %s  N=%d  NNZ=%u  avg_deg=%.1f  K=%d  R=%d\n",
           name, N, nnz, (double)nnz/N, K, R);
    print_deg_stats(csr.indptr, N);

    double ts0 = omp_get_wtime();
    int *perm = make_degree_perm(csr.indptr, N);
    double ts1 = omp_get_wtime();
    printf("  sort: %.2f ms\n", (ts1-ts0)*1000);

    srand(12345);
    float *H = (float*)malloc((size_t)N*K*4);
    float *W = (float*)malloc((size_t)K*K*4);
    for (size_t i = 0; i < (size_t)N*K; i++) H[i]=0.01f*((rand()%200)-100);
    for (int i = 0; i < K*K; i++) W[i]=0.01f*((rand()%200)-100);

    uint16_t *Hb = (uint16_t*)aligned_alloc(64, (size_t)N*K*2);
    uint16_t *Wv = (uint16_t*)aligned_alloc(64, (size_t)KB*NB*16*32*2);
    convert_H_bf16(H, Hb, N, K);
    make_W_vnni(W, Wv, K);
    float *C = (float*)calloc((size_t)N*K, 4);

    /* Warmup */
    tfs_v3k(csr.indptr, csr.indices, Hb, Wv, C, perm, N, R, K);

    double best = 1e30;
    for (int run = 0; run < 5; run++) {
        double t0 = omp_get_wtime();
        tfs_v3k(csr.indptr, csr.indices, Hb, Wv, C, perm, N, R, K);
        double t1 = omp_get_wtime();
        double ms = (t1-t0)*1000;
        printf("  run %d: %.2f ms\n", run, ms);
        if (ms < best) best = ms;
    }
    printf("  BEST: %.2f ms  (K=%d, R=%d)\n\n", best, K, R);

    /* Correctness (first 512 rows) */
    int ck = (N<512)?N:512;
    float *Cr = (float*)calloc((size_t)ck*K, 4);
    ref_twostep(csr.indptr, csr.indices, Hb, W, Cr, ck, K);
    double md=0, mr=0;
    for (size_t i = 0; i < (size_t)ck*K; i++) {
        double d = fabs((double)C[i]-(double)Cr[i]);
        double r = fabs((double)Cr[i]);
        if (d>md)md=d; if(r>mr)mr=r;
    }
    double rel = (mr>0)?md/mr:0;
    printf("  Correctness: rel_err=%.6f %s\n\n", rel, (rel<0.02)?"PASS":"FAIL");

    free(csr.indptr);free(csr.indices);free(H);free(W);
    free(Hb);free(Wv);free(C);free(Cr);free(perm);
}

/* ================================================================ */
int main(int argc, char **argv)
{
    printf("TFS Fusion V3K -- Variable K\n");
    printf("Threads: %d\n\n", omp_get_max_threads());

    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) {
        int Ks[] = {32, 64, 128, 256};
        for (int i = 0; i < 4; i++) selftest(Ks[i]);
        return 0;
    }

    if (argc < 3) {
        fprintf(stderr,
            "Usage:\n"
            "  %s --selftest\n"
            "  %s <dir> <name|ALL> [K=128] [R=64]\n"
            "  %s <dir> <name|ALL> 0       # K sweep 32,64,128,256\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *dir = argv[1], *name = argv[2];
    int K = (argc >= 4) ? atoi(argv[3]) : 128;
    int R = (argc >= 5) ? atoi(argv[4]) : 64;

    if (K != 0 && (K < 32 || K % 32 != 0)) {
        fprintf(stderr, "K must be multiple of 32 (got %d)\n", K);
        return 1;
    }

    const char *all[] = {
        "web-Google","amazon0601","cit-Patents","as-Skitter",
        "soc-Pokec","hollywood-2009","indochina-2004"
    };

    if (strcmp(name, "ALL") == 0) {
        if (K == 0) {
            int Ks[] = {32, 64, 128, 256};
            for (int ki = 0; ki < 4; ki++) {
                printf("########################################\n");
                printf("# K = %d\n", Ks[ki]);
                printf("########################################\n\n");
                for (int mi = 0; mi < 7; mi++)
                    benchmark(dir, all[mi], R, Ks[ki]);
            }
        } else {
            for (int mi = 0; mi < 7; mi++)
                benchmark(dir, all[mi], R, K);
        }
    } else {
        if (K == 0) {
            int Ks[] = {32, 64, 128, 256};
            for (int ki = 0; ki < 4; ki++)
                benchmark(dir, name, R, Ks[ki]);
        } else {
            benchmark(dir, name, R, K);
        }
    }
    return 0;
}
