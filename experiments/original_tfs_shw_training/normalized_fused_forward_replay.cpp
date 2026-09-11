#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
#include <mkl.h>

using bf16 = std::uint16_t;
constexpr int K = 128;
constexpr int TR = 16;

struct Graph {
  int n = 0;
  std::uint64_t nnz = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colidx;
  std::vector<float> scale;
  std::vector<float> x;
};

template <typename T> void read_exact(std::ifstream& f, T* p, std::size_t n) {
  f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n * sizeof(T)));
  if (!f) throw std::runtime_error("truncated fixture");
}

Graph load_fixture(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open fixture: " + path);
  char magic[8]; read_exact(f, magic, 8);
  if (std::memcmp(magic, "TFSFWD01", 8) != 0) throw std::runtime_error("bad fixture magic");
  std::uint64_t n, nnz, k;
  read_exact(f, &n, 1); read_exact(f, &nnz, 1); read_exact(f, &k, 1);
  if (n > std::numeric_limits<int>::max() || k != K) throw std::runtime_error("unsupported fixture shape");
  Graph g; g.n = static_cast<int>(n); g.nnz = nnz;
  g.rowptr.resize(n + 1); g.colidx.resize(nnz); g.scale.resize(n); g.x.resize(n * k);
  read_exact(f, g.rowptr.data(), g.rowptr.size());
  read_exact(f, g.colidx.data(), g.colidx.size());
  read_exact(f, g.scale.data(), g.scale.size());
  read_exact(f, g.x.data(), g.x.size());
  if (g.rowptr.front() != 0 || static_cast<std::uint64_t>(g.rowptr.back()) != nnz)
    throw std::runtime_error("invalid CSR rowptr");
  return g;
}

static inline bf16 to_bf16(float x) {
  std::uint32_t u; std::memcpy(&u, &x, 4);
  const std::uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16>(u >> 16);
}
static inline float from_bf16(bf16 x) {
  std::uint32_t u = static_cast<std::uint32_t>(x) << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

struct alignas(64) TileCfg {
  std::uint8_t palette, start_row, reserved[14];
  std::uint16_t colsb[16];
  std::uint8_t rows[16];
};

void setup_tiles(TileCfg& cfg, int d) {
  std::memset(&cfg, 0, sizeof(cfg)); cfg.palette = 1;
  (void)d;
  for (int t = 0; t < 8; ++t) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
}

std::vector<int> degree_perm(const Graph& g) {
  std::vector<int> p(g.n); std::iota(p.begin(), p.end(), 0);
  std::stable_sort(p.begin(), p.end(), [&](int a, int b) {
    return g.rowptr[a + 1] - g.rowptr[a] < g.rowptr[b + 1] - g.rowptr[b];
  });
  return p;
}

void make_hs(const Graph& g, std::vector<bf16>& hs, int threads) {
  #pragma omp parallel for num_threads(threads) schedule(static)
  for (int i = 0; i < g.n; ++i)
    for (int k = 0; k < K; ++k)
      hs[static_cast<std::size_t>(i) * K + k] =
          to_bf16(g.x[static_cast<std::size_t>(i) * K + k] * g.scale[i]);
}

void make_weight(int d, int dp, std::vector<float>& w, std::vector<float>& bias) {
  w.assign(static_cast<std::size_t>(K) * dp, 0.0f); bias.resize(d);
  std::uint32_t state = 123456789u;
  auto rnd = [&]() { state = state * 1664525u + 1013904223u; return (int(state >> 8) % 2001 - 1000) * 1e-4f; };
  for (int k = 0; k < K; ++k) for (int j = 0; j < d; ++j) w[static_cast<std::size_t>(k) * dp + j] = rnd();
  for (int j = 0; j < d; ++j) bias[j] = rnd();
}

void pack_w_rowmajor(const std::vector<float>& w, int d, int dp, std::vector<bf16>& wb) {
  wb.resize(static_cast<std::size_t>(K) * d);
  for (int k = 0; k < K; ++k) for (int j = 0; j < d; ++j)
    wb[static_cast<std::size_t>(k) * d + j] = to_bf16(w[static_cast<std::size_t>(k) * dp + j]);
}

void pack_w_vnni(const std::vector<float>& w, int dp, std::vector<bf16>& wv) {
  const int nb = dp / 16;
  wv.resize(static_cast<std::size_t>(4) * nb * 16 * 32);
  for (int kb = 0; kb < 4; ++kb) for (int ob = 0; ob < nb; ++ob)
    for (int kp = 0; kp < 16; ++kp) for (int j = 0; j < 16; ++j) {
      const int k0 = kb * 32 + kp * 2, col = ob * 16 + j;
      const std::size_t at = ((static_cast<std::size_t>(kb) * nb + ob) * 16 + kp) * 32 + j * 2;
      wv[at] = to_bf16(w[static_cast<std::size_t>(k0) * dp + col]);
      wv[at + 1] = to_bf16(w[static_cast<std::size_t>(k0 + 1) * dp + col]);
    }
}

void pack_w_tail40(const std::vector<float>& w, int dp, std::vector<bf16>& tail) {
  tail.resize(static_cast<std::size_t>(4) * 16 * 16);
  for (int kb=0;kb<4;++kb) for(int kp=0;kp<16;++kp) for(int j=0;j<8;++j) {
    const int k0=kb*32+kp*2, col=32+j; const std::size_t at=(static_cast<std::size_t>(kb)*16+kp)*16+j*2;
    tail[at]=to_bf16(w[static_cast<std::size_t>(k0)*dp+col]);
    tail[at+1]=to_bf16(w[static_cast<std::size_t>(k0+1)*dp+col]);
  }
}

inline __m512 load_bf16_16(const bf16* p) {
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}

void pull_to_bf16(const Graph& g, const bf16* hs, bf16* t, int threads) {
  #pragma omp parallel for num_threads(threads) schedule(dynamic,64)
  for (int i = 0; i < g.n; ++i) for (int k = 0; k < K; k += 16) {
    __m512 sum = load_bf16_16(hs + static_cast<std::size_t>(i) * K + k);
    for (std::int64_t e = g.rowptr[i]; e < g.rowptr[i + 1]; ++e)
      sum = _mm512_add_ps(sum, load_bf16_16(hs + static_cast<std::size_t>(g.colidx[e]) * K + k));
    alignas(64) float tmp[16]; _mm512_store_ps(tmp, sum);
    for (int q = 0; q < 16; ++q) t[static_cast<std::size_t>(i) * K + k + q] = to_bf16(tmp[q]);
  }
}

void epilogue(const Graph& g, float* y, int d, const std::vector<float>& bias, int threads) {
  #pragma omp parallel for num_threads(threads) schedule(static)
  for (int i = 0; i < g.n; ++i)
    for (int j = 0; j < d; ++j) y[static_cast<std::size_t>(i) * d + j] = y[static_cast<std::size_t>(i) * d + j] * g.scale[i] + bias[j];
}

void pull_fp32_epilogue(const Graph& g, const float* u, float* y, int d,
                        const std::vector<float>& bias, int threads) {
  #pragma omp parallel for num_threads(threads) schedule(dynamic,64)
  for (int i = 0; i < g.n; ++i) {
    int j = 0;
    for (; j + 16 <= d; j += 16) {
      __m512 sum = _mm512_loadu_ps(u + static_cast<std::size_t>(i) * d + j);
      for (std::int64_t e = g.rowptr[i]; e < g.rowptr[i + 1]; ++e)
        sum = _mm512_add_ps(sum, _mm512_loadu_ps(u + static_cast<std::size_t>(g.colidx[e]) * d + j));
      sum = _mm512_fmadd_ps(sum, _mm512_set1_ps(g.scale[i]), _mm512_loadu_ps(bias.data() + j));
      _mm512_storeu_ps(y + static_cast<std::size_t>(i) * d + j, sum);
    }
    for (; j < d; ++j) {
      float sum = u[static_cast<std::size_t>(i) * d + j];
      for (std::int64_t e = g.rowptr[i]; e < g.rowptr[i + 1]; ++e) sum += u[static_cast<std::size_t>(g.colidx[e]) * d + j];
      y[static_cast<std::size_t>(i) * d + j] = sum * g.scale[i] + bias[j];
    }
  }
}

void fused_amx(const Graph& g, const bf16* hs, const bf16* wv, const bf16* tail40, float* y,
               const int* perm, int d, int dp, int threads, int group_rows,
               const std::vector<float>& bias) {
  const int nb = dp / 16, np = dp / 64;
  (void)tail40;
  #pragma omp parallel num_threads(threads)
  {
    if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) throw std::runtime_error("AMX permission failed");
    TileCfg cfg; setup_tiles(cfg,d); _tile_loadconfig(&cfg);
    alignas(64) bf16 hbuf[TR * K]; alignas(64) float ctmp[TR * 16];
    std::int64_t base[TR], deg[TR]; int rows[TR];
    #pragma omp for schedule(dynamic,1)
    for (int rg = 0; rg < g.n; rg += group_rows) {
      const int end = std::min(g.n, rg + group_rows);
      for (int ii = rg; ii < end; ii += TR) {
        const int batch = std::min(TR, end - ii); int max_steps = 0;
        for (int r = 0; r < batch; ++r) {
          rows[r] = perm[ii + r]; base[r] = g.rowptr[rows[r]];
          deg[r] = g.rowptr[rows[r] + 1] - base[r]; max_steps = std::max(max_steps, int(deg[r] + 1));
        }
        for (int pass = 0; pass < np; ++pass) {
          _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
          std::memset(hbuf, 0, sizeof(hbuf)); int active_from = 0;
          for (int step = 0; step < max_steps; ++step) {
            while (active_from < batch && step > deg[active_from]) { std::memset(hbuf + active_from * K, 0, K * sizeof(bf16)); ++active_from; }
            if (active_from == batch) break;
            for (int r = active_from; r < batch; ++r) {
              const int src = step == 0 ? rows[r] : g.colidx[base[r] + step - 1];
              std::memcpy(hbuf + r * K, hs + static_cast<std::size_t>(src) * K, K * sizeof(bf16));
            }
            for (int kb = 0; kb < 4; ++kb) {
              _tile_loadd(4, reinterpret_cast<const char*>(hbuf) + kb * 64, K * 2);
              const int ob0 = pass * 4;
              const bf16* b0 = wv + (static_cast<std::size_t>(kb) * nb + ob0) * 16 * 32;
              _tile_loadd(5, b0, 64); _tile_loadd(6, b0 + 16 * 32, 64);
              _tile_dpbf16ps(0, 4, 5); _tile_dpbf16ps(1, 4, 6);
              if (d == 40) {
                _tile_loadd(5, b0 + 2 * 16 * 32, 64);
                _tile_dpbf16ps(2, 4, 5);
              } else {
                _tile_loadd(5, b0 + 2 * 16 * 32, 64); _tile_loadd(6, b0 + 3 * 16 * 32, 64);
                _tile_dpbf16ps(2, 4, 5); _tile_dpbf16ps(3, 4, 6);
              }
            }
          }
          auto scatter = [&](int q, int col) {
            for (int r = 0; r < batch; ++r) for (int j = 0; j < 16 && col + j < d; ++j)
              y[static_cast<std::size_t>(rows[r]) * d + col + j] = ctmp[r * 16 + j] * g.scale[rows[r]] + bias[col + j];
          };
          _tile_stored(0, ctmp, 64); scatter(0, pass * 64);
          _tile_stored(1, ctmp, 64); scatter(1, pass * 64 + 16);
          _tile_stored(2, ctmp, 64); scatter(2, pass * 64 + 32);
          if (d != 40) { _tile_stored(3, ctmp, 64); scatter(3, pass * 64 + 48); }
        }
      }
    }
    _tile_release();
  }
}

enum class ReplayPack { Memcpy, Avx512, KBlock };

static inline void copy_source_row(const bf16* src, bf16* dst, ReplayPack mode) {
  if (mode == ReplayPack::Memcpy) {
    std::memcpy(dst, src, K * sizeof(bf16));
    return;
  }
  for (int kb = 0; kb < 4; ++kb) {
    const __m512i z = _mm512_loadu_si512(reinterpret_cast<const void*>(src + kb * 32));
    _mm512_store_si512(reinterpret_cast<void*>(dst + kb * 32), z);
  }
}

static inline const bf16* replay_ap(const bf16* replay, bool kblock, int step, int kb) {
  return kblock ? replay + (static_cast<std::size_t>(step) * 4 + kb) * TR * 32
                : replay + static_cast<std::size_t>(step) * TR * K + kb * 32;
}

static inline void add_tile_values(const float* tile, float* out, int col) {
  for (int r = 0; r < TR; ++r) {
    float* dst = out + static_cast<std::size_t>(r) * 128 + col;
    _mm512_store_ps(dst, _mm512_add_ps(_mm512_load_ps(dst), _mm512_load_ps(tile + r * 16)));
  }
}

#define DEFINE_STORE_TILE(TID) \
static inline void store_tile##TID(float* out, int col, bool first) { \
  if (first) { \
    _tile_stored(TID, out + col, 128 * sizeof(float)); \
  } else { \
    alignas(64) float tile[TR * 16]; \
    _tile_stored(TID, tile, 64); \
    add_tile_values(tile, out, col); \
  } \
}
DEFINE_STORE_TILE(0)
DEFINE_STORE_TILE(1)
DEFINE_STORE_TILE(2)
DEFINE_STORE_TILE(3)
DEFINE_STORE_TILE(4)
DEFINE_STORE_TILE(5)
#undef DEFINE_STORE_TILE

static inline void replay_pass4(const bf16* replay, bool kblock, int chunk_steps,
                                const bf16* wv, int nb, int ob0, int valid, float* out, bool first) {
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  for (int step = 0; step < chunk_steps; ++step) for (int kb = 0; kb < 4; ++kb) {
    _tile_loadd(4, replay_ap(replay,kblock,step,kb), kblock ? 64 : K * 2);
    const bf16* bp = wv + (static_cast<std::size_t>(kb) * nb + ob0) * 16 * 32;
    _tile_loadd(5,bp,64); _tile_dpbf16ps(0,4,5);
    _tile_loadd(6,bp+16*32,64); _tile_dpbf16ps(1,4,6);
    if (valid >= 3) { _tile_loadd(5,bp+2*16*32,64); _tile_dpbf16ps(2,4,5); }
    if (valid >= 4) { _tile_loadd(6,bp+3*16*32,64); _tile_dpbf16ps(3,4,6); }
  }
  store_tile0(out,ob0*16,first); store_tile1(out,(ob0+1)*16,first);
  if (valid >= 3) store_tile2(out,(ob0+2)*16,first);
  if (valid >= 4) store_tile3(out,(ob0+3)*16,first);
}

static inline void replay_pass5(const bf16* replay, bool kblock, int chunk_steps,
                                const bf16* wv, int nb, int ob0, float* out, bool first) {
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); _tile_zero(4);
  for (int step=0;step<chunk_steps;++step) for(int kb=0;kb<4;++kb) {
    _tile_loadd(5,replay_ap(replay,kblock,step,kb),kblock?64:K*2);
    const bf16* bp=wv+(static_cast<std::size_t>(kb)*nb+ob0)*16*32;
    _tile_loadd(6,bp,64); _tile_dpbf16ps(0,5,6);
    _tile_loadd(7,bp+16*32,64); _tile_dpbf16ps(1,5,7);
    _tile_loadd(6,bp+2*16*32,64); _tile_dpbf16ps(2,5,6);
    _tile_loadd(7,bp+3*16*32,64); _tile_dpbf16ps(3,5,7);
    _tile_loadd(6,bp+4*16*32,64); _tile_dpbf16ps(4,5,6);
  }
  store_tile0(out,ob0*16,first); store_tile1(out,(ob0+1)*16,first);
  store_tile2(out,(ob0+2)*16,first); store_tile3(out,(ob0+3)*16,first);
  store_tile4(out,(ob0+4)*16,first);
}

static inline void replay_pass6(const bf16* replay, bool kblock, int chunk_steps,
                                const bf16* wv, int nb, int ob0, float* out, bool first) {
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); _tile_zero(4); _tile_zero(5);
  for (int step=0;step<chunk_steps;++step) for(int kb=0;kb<4;++kb) {
    _tile_loadd(6,replay_ap(replay,kblock,step,kb),kblock?64:K*2);
    const bf16* bp=wv+(static_cast<std::size_t>(kb)*nb+ob0)*16*32;
    _tile_loadd(7,bp,64); _tile_dpbf16ps(0,6,7);
    _tile_loadd(7,bp+16*32,64); _tile_dpbf16ps(1,6,7);
    _tile_loadd(7,bp+2*16*32,64); _tile_dpbf16ps(2,6,7);
    _tile_loadd(7,bp+3*16*32,64); _tile_dpbf16ps(3,6,7);
    _tile_loadd(7,bp+4*16*32,64); _tile_dpbf16ps(4,6,7);
    _tile_loadd(7,bp+5*16*32,64); _tile_dpbf16ps(5,6,7);
  }
  store_tile0(out,ob0*16,first); store_tile1(out,(ob0+1)*16,first);
  store_tile2(out,(ob0+2)*16,first); store_tile3(out,(ob0+3)*16,first);
  store_tile4(out,(ob0+4)*16,first); store_tile5(out,(ob0+5)*16,first);
}

void fused_amx_replay128(const Graph& g, const bf16* hs, const bf16* wv, float* y,
                         const int* perm, int threads, int group_rows, int replay_steps,
                         int ctiles, ReplayPack pack_mode, const std::vector<float>& bias) {
  if (replay_steps <= 0 || replay_steps > 32 || (replay_steps & (replay_steps - 1)) != 0)
    throw std::runtime_error("replay steps must be one of 1,2,4,8,16,32");
  if (ctiles != 4 && ctiles != 5 && ctiles != 6)
    throw std::runtime_error("ctiles must be 4, 5, or 6");
  constexpr int nb = 8;
  #pragma omp parallel num_threads(threads)
  {
    if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) throw std::runtime_error("AMX permission failed");
    TileCfg cfg; setup_tiles(cfg,128); _tile_loadconfig(&cfg);
    const bool kblock = pack_mode == ReplayPack::KBlock;
    thread_local std::vector<bf16> replay;
    replay.resize(static_cast<std::size_t>(replay_steps) * TR * K);
    alignas(64) float accum[TR * 128];
    std::int64_t base[TR], deg[TR]; int rows[TR];
    #pragma omp for schedule(dynamic,1)
    for (int rg = 0; rg < g.n; rg += group_rows) {
      const int end = std::min(g.n, rg + group_rows);
      for (int ii = rg; ii < end; ii += TR) {
        const int batch = std::min(TR, end - ii); int max_steps = 0;
        for (int r = 0; r < batch; ++r) {
          rows[r] = perm[ii + r]; base[r] = g.rowptr[rows[r]];
          deg[r] = g.rowptr[rows[r] + 1] - base[r];
          max_steps = std::max(max_steps, int(deg[r] + 1));
        }
        for (int cs = 0; cs < max_steps; cs += replay_steps) {
          const int chunk = std::min(replay_steps, max_steps - cs);
          std::memset(replay.data(), 0, static_cast<std::size_t>(chunk) * TR * K * sizeof(bf16));
          for (int ls = 0; ls < chunk; ++ls) {
            const int step = cs + ls;
            for (int r = 0; r < batch; ++r) {
              if (step > deg[r]) continue;
              const int src = step == 0 ? rows[r] : g.colidx[base[r] + step - 1];
              const bf16* srcp = hs + static_cast<std::size_t>(src) * K;
              if (!kblock) {
                copy_source_row(srcp, replay.data() + (static_cast<std::size_t>(ls) * TR + r) * K, pack_mode);
              } else {
                for (int kb = 0; kb < 4; ++kb) {
                  bf16* dst = replay.data() + (static_cast<std::size_t>(ls) * 4 + kb) * TR * 32 + r * 32;
                  const __m512i z = _mm512_loadu_si512(reinterpret_cast<const void*>(srcp + kb * 32));
                  _mm512_store_si512(reinterpret_cast<void*>(dst), z);
                }
              }
            }
          }
          const bool first = cs == 0;
          if (ctiles == 4) {
            replay_pass4(replay.data(), kblock, chunk, wv, nb, 0, 4, accum, first);
            replay_pass4(replay.data(), kblock, chunk, wv, nb, 4, 4, accum, first);
          } else if (ctiles == 5) {
            replay_pass5(replay.data(), kblock, chunk, wv, nb, 0, accum, first);
            replay_pass4(replay.data(), kblock, chunk, wv, nb, 5, 3, accum, first);
          } else {
            replay_pass6(replay.data(), kblock, chunk, wv, nb, 0, accum, first);
            replay_pass4(replay.data(), kblock, chunk, wv, nb, 6, 2, accum, first);
          }
        }
        for (int r = 0; r < batch; ++r) {
          const float scale = g.scale[rows[r]];
          for (int j = 0; j < 128; ++j)
            y[static_cast<std::size_t>(rows[r]) * 128 + j] = accum[r * 128 + j] * scale + bias[j];
        }
      }
    }
    _tile_release();
  }
}

struct Times { double prep=0, sparse=0, dense=0, ep=0, total=0; };
struct Workspace {
  std::vector<bf16> hs, wb, t, wv, tail40;
  std::vector<float> u;
  Workspace(const Graph& g, int d, int dp)
      : hs(static_cast<std::size_t>(g.n)*K), wb(static_cast<std::size_t>(K)*d),
        t(static_cast<std::size_t>(g.n)*K), wv(static_cast<std::size_t>(4)*(dp/16)*16*32),
        tail40(d==40 ? static_cast<std::size_t>(4)*16*16 : 0),
        u(static_cast<std::size_t>(g.n)*d) {}
};
double ms() { return omp_get_wtime() * 1000.0; }

Times run_aggregate(const Graph& g, const std::vector<float>& w, const std::vector<float>& bias,
                    int d, int dp, int threads, Workspace& z, std::vector<float>& y) {
  const double t0=ms(); make_hs(g,z.hs,threads); pack_w_rowmajor(w,d,dp,z.wb); const double t1=ms();
  pull_to_bf16(g,z.hs.data(),z.t.data(),threads); const double t2=ms();
  mkl_set_num_threads_local(threads);
  cblas_gemm_bf16bf16f32(CblasRowMajor,CblasNoTrans,CblasNoTrans,g.n,d,K,1.0f,
      reinterpret_cast<const MKL_BF16*>(z.t.data()),K,reinterpret_cast<const MKL_BF16*>(z.wb.data()),d,0.0f,y.data(),d);
  const double t3=ms(); epilogue(g,y.data(),d,bias,threads); const double t4=ms();
  return {t1-t0,t2-t1,t3-t2,t4-t3,t4-t0};
}

Times run_transform(const Graph& g, const std::vector<float>& w, const std::vector<float>& bias,
                    int d, int dp, int threads, Workspace& z, std::vector<float>& y) {
  const double t0=ms(); make_hs(g,z.hs,threads); pack_w_rowmajor(w,d,dp,z.wb); const double t1=ms(); mkl_set_num_threads_local(threads);
  cblas_gemm_bf16bf16f32(CblasRowMajor,CblasNoTrans,CblasNoTrans,g.n,d,K,1.0f,
      reinterpret_cast<const MKL_BF16*>(z.hs.data()),K,reinterpret_cast<const MKL_BF16*>(z.wb.data()),d,0.0f,z.u.data(),d);
  const double t2=ms(); pull_fp32_epilogue(g,z.u.data(),y.data(),d,bias,threads); const double t3=ms();
  return {t1-t0,t3-t2,t2-t1,0,t3-t0};
}

Times run_fused(const Graph& g, const std::vector<float>& w, const std::vector<float>& bias,
                const std::vector<int>& perm, int d, int dp, int threads, int group_rows,
                Workspace& z, std::vector<float>& y, const std::string& variant,
                int replay_steps, int ctiles, ReplayPack pack_mode) {
  const double t0=ms(); make_hs(g,z.hs,threads); pack_w_vnni(w,dp,z.wv); const double t1=ms();
  if (variant == "legacy")
    fused_amx(g,z.hs.data(),z.wv.data(),nullptr,y.data(),perm.data(),d,dp,threads,group_rows,bias);
  else {
    if (d != 128) throw std::runtime_error("replay variants currently require d=128");
    fused_amx_replay128(g,z.hs.data(),z.wv.data(),y.data(),perm.data(),threads,group_rows,
                        replay_steps,ctiles,pack_mode,bias);
  }
  const double t2=ms();
  return {t1-t0,t2-t1,0,0,t2-t0};
}

void error(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
  double maxa=0, sum2=0, ref2=0;
  for (std::size_t i=0;i<got.size();++i) { double e=double(got[i])-ref[i]; maxa=std::max(maxa,std::abs(e)); sum2+=e*e; ref2+=double(ref[i])*ref[i]; }
  std::cout << "ERROR,path="<<name<<",max_abs="<<maxa<<",rel_l2="<<std::sqrt(sum2/ref2)<<"\n";
}

int main(int argc,char**argv) {
  std::string fixture,variant="legacy",pack="memcpy"; int d=128,threads=1,repeats=7,warmups=2,group_rows=64,replay_steps=16,ctiles=4;
  for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing value");std::string k=argv[i],v=argv[i+1];
    if(k=="--fixture")fixture=v;else if(k=="--d")d=std::stoi(v);else if(k=="--threads")threads=std::stoi(v);
    else if(k=="--repeats")repeats=std::stoi(v);else if(k=="--warmups")warmups=std::stoi(v);else if(k=="--group-rows")group_rows=std::stoi(v);
    else if(k=="--variant")variant=v;else if(k=="--replay-steps")replay_steps=std::stoi(v);else if(k=="--ctiles")ctiles=std::stoi(v);else if(k=="--pack")pack=v;
    else throw std::runtime_error("unknown arg "+k);}
  if(fixture.empty() || (d!=40&&d!=128))throw std::runtime_error("need --fixture and d=40|128");
  ReplayPack pack_mode = pack=="memcpy" ? ReplayPack::Memcpy : pack=="avx" ? ReplayPack::Avx512 : pack=="kblock" ? ReplayPack::KBlock : throw std::runtime_error("pack must be memcpy|avx|kblock");
  omp_set_dynamic(0); Graph g=load_fixture(fixture); const int dp=((d+63)/64)*64;
  const double s0=ms(); auto perm=degree_perm(g); const double sort_ms=ms()-s0;
  std::vector<float>w,bias;make_weight(d,dp,w,bias);std::vector<float> ya(static_cast<std::size_t>(g.n)*d),yt(ya.size()),yf(ya.size());
  Workspace za(g,d,dp),zt(g,d,dp),zf(g,d,dp);
  std::cout<<"META,n="<<g.n<<",nnz="<<g.nnz<<",d="<<d<<",threads="<<threads<<",sort_ms="<<sort_ms<<",group_rows="<<group_rows<<",variant="<<variant<<",replay_steps="<<replay_steps<<",ctiles="<<ctiles<<",pack="<<pack<<"\n";
  run_transform(g,w,bias,d,dp,threads,zt,yt);run_aggregate(g,w,bias,d,dp,threads,za,ya);run_fused(g,w,bias,perm,d,dp,threads,group_rows,zf,yf,variant,replay_steps,ctiles,pack_mode);
  error("aggregate_first",ya,yt);error("fused_amx",yf,yt);
  for(int r=-warmups;r<repeats;++r){auto a=run_aggregate(g,w,bias,d,dp,threads,za,ya);auto t=run_transform(g,w,bias,d,dp,threads,zt,yt);auto f=run_fused(g,w,bias,perm,d,dp,threads,group_rows,zf,yf,variant,replay_steps,ctiles,pack_mode);
    if(r>=0){auto out=[&](const char*p,const Times&z){std::cout<<"RESULT,path="<<p<<",repeat="<<r<<",prep_ms="<<z.prep<<",sparse_ms="<<z.sparse<<",dense_ms="<<z.dense<<",epilogue_ms="<<z.ep<<",total_ms="<<z.total<<"\n";};out("aggregate_first",a);out("transform_first",t);out("fused_amx",f);}}
}
