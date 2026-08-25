#include <mutex>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "../../amx/backward_v2_kernels.h"
#include "../products_saved_t_20260812/saved_t_extension.h"

namespace bv2 = tfs::backward_v2;
using bv2::bf16;

namespace {

struct ColidxValidationEntry {
  const std::int64_t* identity;
  std::int64_t count;
  bool valid;
};

// The validation cache is process-global rather than a local static inside
// an inline helper.  Intel's optimizer may clone inline call sites, which can
// otherwise turn the supposedly one-time ``auto`` check back into a full CSR
// scan on every invocation.  The cache is keyed by CSR pointer and length so
// one process may safely serve more than one immutable graph.
std::mutex g_colidx_validation_mu;
std::vector<ColidxValidationEntry> g_colidx_validation_cache;

inline bool profile_clock_enabled() {
  // The authority path leaves native profiling off.  Returning immediately
  // here removes thousands of steady_clock calls from every backward while
  // preserving the exact diagnostic timestamps when profiling is requested.
  static const bool enabled = [] {
    const char* v = std::getenv("TFS_INTERNAL_PROFILE");
    return v != nullptr && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

inline double now_ms() {
  if (!profile_clock_enabled()) return 0.0;
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
      clock::now().time_since_epoch()).count();
}

inline bool internal_profile_enabled() {
  const char* v = std::getenv("TFS_INTERNAL_PROFILE");
  return v != nullptr && std::strcmp(v, "0") != 0;
}

inline bool numa_workspace_profile_enabled() {
  const char* v = std::getenv("TFS_PROFILE_NUMA_WORKSPACE");
  return v != nullptr && std::strcmp(v, "0") != 0 &&
      std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

inline double steady_ms_unconditional() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
      clock::now().time_since_epoch()).count();
}

inline bool experiment_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::strcmp(v, "0") != 0;
}

inline bool highd_adaptive_flag(const char* name, bool auto_value) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0' || std::strcmp(v, "auto") == 0)
    return auto_value;
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
      std::strcmp(v, "false") == 0 || std::strcmp(v, "no") == 0)
    return false;
  return true;
}

// Formal r5 controls use auto|off|on.  Keep the legacy experiment flags
// above intact for reproducibility, while allowing the standard planner to
// enable a generic path without dataset-specific names.
inline bool formal_mode_enabled(const char* name, bool auto_value) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0' || std::strcmp(v, "auto") == 0)
    return auto_value;
  if (std::strcmp(v, "on") == 0 || std::strcmp(v, "1") == 0 ||
      std::strcmp(v, "true") == 0)
    return true;
  if (std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0 ||
      std::strcmp(v, "false") == 0)
    return false;
  TORCH_CHECK(false, name, " must be auto|off|on, got ", v);
  return false;
}

bool formal_colidx_enabled(bool legacy, const std::int64_t* ci,
                           std::int64_t count) {
  const char* v = std::getenv("TFS_COLIDX");
  if (v == nullptr || *v == '\0') return legacy;
  if (std::strcmp(v, "int64") == 0 || std::strcmp(v, "off") == 0 ||
      std::strcmp(v, "0") == 0)
    return false;
  if (std::strcmp(v, "int32") == 0 || std::strcmp(v, "on") == 0 ||
      std::strcmp(v, "1") == 0)
    return true;
  TORCH_CHECK(std::strcmp(v, "auto") == 0,
              "TFS_COLIDX must be auto|int32|int64, got ", v);
  // ``auto`` must validate a graph once, not rescan the complete CSR on
  // every forward/backward call.  Products has 123M+ entries, so the old
  // per-call validation added roughly 100 ms to every layer invocation and
  // erased the intended int32 CSR benefit.  The graph tensors are immutable
  // for the full-batch training contract; pointer+length is therefore the
  // same identity used by the conversion workspace below.
  std::lock_guard<std::mutex> lock(g_colidx_validation_mu);
  for (const auto& entry : g_colidx_validation_cache) {
    if (entry.identity == ci && entry.count == count) return entry.valid;
  }
  bool valid = true;
  for (std::int64_t e = 0; e < count; ++e) {
    if (ci[e] < 0 || ci[e] > INT32_MAX) {
      valid = false;
      break;
    }
  }
  g_colidx_validation_cache.push_back({ci, count, valid});
  return valid;
}

inline int cpu_sysfs_int(int cpu, const char* leaf, int fallback = -1) {
  char path[256];
  std::snprintf(path, sizeof(path),
                "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, leaf);
  FILE* f = std::fopen(path, "r");
  if (f == nullptr) return fallback;
  int value = fallback;
  if (std::fscanf(f, "%d", &value) != 1) value = fallback;
  std::fclose(f);
  return value;
}

inline int cpu_numa_node(int cpu) {
  char path[256];
  for (int node = 0; node < 256; ++node) {
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/node%d", cpu, node);
    if (::access(path, F_OK) == 0) return node;
  }
  return -1;
}

std::vector<int> parse_cpu_list(const char* text) {
  std::vector<int> cpus;
  if (text == nullptr || *text == '\0') return cpus;
  const char* p = text;
  while (*p != '\0') {
    char* end = nullptr;
    const long first = std::strtol(p, &end, 10);
    TORCH_CHECK(end != p && first >= 0 && first < CPU_SETSIZE,
                "invalid TFS_WORKER_CPUS entry: ", text);
    long last = first;
    p = end;
    if (*p == '-') {
      ++p;
      last = std::strtol(p, &end, 10);
      TORCH_CHECK(end != p && last >= first && last < CPU_SETSIZE,
                  "invalid TFS_WORKER_CPUS range: ", text);
      p = end;
    }
    for (long cpu = first; cpu <= last; ++cpu) cpus.push_back(static_cast<int>(cpu));
    if (*p == ',') ++p;
    else TORCH_CHECK(*p == '\0', "invalid TFS_WORKER_CPUS separator: ", text);
  }
  std::sort(cpus.begin(), cpus.end());
  cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
  return cpus;
}

// Derive the reduction partition from the actual worker CPU list instead of
// assuming four NUMA groups with eight workers each.  The reduction remains
// deterministic (workers are counted in sorted CPU order), while this helper
// makes 1/2/4/8/16/32-thread runs use the topology they were launched with.
inline int runtime_per_numa(int threads) {
  std::vector<int> cpus = parse_cpu_list(std::getenv("TFS_WORKER_CPUS"));
  if (cpus.empty()) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
  }
  if (cpus.empty()) return std::max(1, std::min(8, threads));
  std::vector<int> counts(256, 0);
  const int active = std::min(threads, static_cast<int>(cpus.size()));
  int max_count = 0;
  for (int tid = 0; tid < active; ++tid) {
    const int node = cpu_numa_node(cpus[tid]);
    const int slot = node < 0 ? 0 : std::min(node, 255);
    max_count = std::max(max_count, ++counts[slot]);
  }
  return std::max(1, max_count);
}

inline int runtime_reduce_group(int threads) {
  return std::max(1, std::min(4, runtime_per_numa(threads)));
}

// The worker pool assigns the frozen CPU list in sorted order.  Keep the
// replica layout consistent with the scheduler/reduction layout: a contiguous
// group of ``per_numa`` logical workers owns one replica.  This deliberately
// avoids assuming that Linux NUMA node IDs are dense (the runtime only needs a
// stable ownership slot, not the OS node number itself).
inline int runtime_numa_count(int threads) {
  const int per_numa = runtime_per_numa(threads);
  return std::max(1, (threads + per_numa - 1) / per_numa);
}

inline int runtime_worker_numa_slot(int tid, int threads) {
  return std::min(runtime_numa_count(threads) - 1,
                  tid / std::max(1, runtime_per_numa(threads)));
}

inline int hs_replica_count_for_threads(int threads) {
  const int available = runtime_numa_count(threads);
  const char* text = std::getenv("TFS_HS_REPLICA_MAX");
  if (text == nullptr || *text == '\0') return available;
  char* end = nullptr;
  const long requested = std::strtol(text, &end, 10);
  TORCH_CHECK(end != text && *end == '\0' && requested >= 1,
              "TFS_HS_REPLICA_MAX must be a positive integer");
  return std::max(1, std::min(available, static_cast<int>(requested)));
}

inline bool numa_first_touch_enabled(int threads) {
  const char* v = std::getenv("TFS_NUMA_FIRST_TOUCH");
  if (v == nullptr || *v == '\0' || std::strcmp(v, "auto") == 0)
    return runtime_per_numa(threads) > 1;
  if (std::strcmp(v, "on") == 0 || std::strcmp(v, "1") == 0 ||
      std::strcmp(v, "true") == 0)
    return true;
  if (std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0 ||
      std::strcmp(v, "false") == 0)
    return false;
  TORCH_CHECK(false, "TFS_NUMA_FIRST_TOUCH must be auto|off|on, got ", v);
  return false;
}

inline bool locality_schedule_enabled(int n, int threads) {
  // The automatic gate is deliberately conservative: constructing source
  // signatures is worthwhile only once there are enough panels and workers
  // for cross-panel cache reuse to matter.  ``on`` remains available for
  // isolated scheduler experiments and ``off`` is the exact legacy schedule.
  return formal_mode_enabled("TFS_LOCALITY_SCHEDULE",
                             n >= 16384 && threads >= 4);
}

// The extension deliberately does not use compiler-side OpenMP: Intel's
// OpenMP runtime and PyTorch's bundled GNU runtime must not coexist in one
// process.  at::parallel_for is header-inline and becomes serial when this
// translation unit is compiled without OpenMP, so use a small persistent
// standard-C++ pool instead.  Logical worker IDs remain stable and therefore
// preserve panel ownership and deterministic dW reduction order.
class PersistentWorkerPool {
 public:
  static PersistentWorkerPool& instance(int requested) {
    // The pool is process-global.  Its capacity must therefore be independent
    // of the first kernel invocation: a 1-thread warm-up may legitimately be
    // followed by a 32-thread measurement in the same interpreter.
    // Authority execution has a validated hard cap of 32 workers.
    (void)requested;
    static PersistentWorkerPool pool(32);
    return pool;
  }

  template <class Fn> void run(int active, Fn&& fn) {
    TORCH_CHECK(active >= 1 && active <= static_cast<int>(workers_.size()),
                "invalid persistent worker count");
    TORCH_CHECK(allowed_cpus_.empty() || active <= static_cast<int>(allowed_cpus_.size()),
                "persistent workers exceed frozen CPU allocation: workers=", active,
                " cpus=", allowed_cpus_.size());
    if (active == 1) {
      fn(0);
      return;
    }
    std::unique_lock<std::mutex> submit_lock(submit_mu_);
    pthread_mutex_lock(&mu_);
    active_ = active;
    completed_ = 0;
    completed_atomic_.store(0, std::memory_order_relaxed);
    failure_ = nullptr;
    task_ = std::forward<Fn>(fn);
    generation_.fetch_add(1, std::memory_order_release);
    pthread_cond_broadcast(&start_cv_);
    if (e8_atomic_done_) {
      while (completed_atomic_.load(std::memory_order_acquire) != active_)
        pthread_cond_wait(&done_cv_, &mu_);
    } else {
      while (completed_ != active_) pthread_cond_wait(&done_cv_, &mu_);
    }
    auto failure = failure_;
    task_ = nullptr;
    pthread_mutex_unlock(&mu_);
    if (failure) std::rethrow_exception(failure);
  }

  ~PersistentWorkerPool() {
    pthread_mutex_lock(&mu_);
    stop_ = true;
    generation_.fetch_add(1, std::memory_order_release);
    pthread_cond_broadcast(&start_cv_);
    pthread_mutex_unlock(&mu_);
    for (auto& worker : workers_) if (worker.joinable()) worker.join();
    pthread_cond_destroy(&done_cv_);
    pthread_cond_destroy(&start_cv_);
    pthread_mutex_destroy(&mu_);
  }

 private:
  explicit PersistentWorkerPool(int capacity) {
    pthread_mutex_init(&mu_, nullptr);
    pthread_cond_init(&start_cv_, nullptr);
    pthread_cond_init(&done_cv_, nullptr);
    allowed_cpus_ = parse_cpu_list(std::getenv("TFS_WORKER_CPUS"));
    if (allowed_cpus_.empty()) {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
          if (CPU_ISSET(cpu, &mask)) allowed_cpus_.push_back(cpu);
      }
    }
    e8_atomic_done_ = experiment_flag("TFS_GLUE_E8_ATOMIC_DONE");
    e8_spin_ = experiment_flag("TFS_GLUE_E8_SPIN");
    workers_.reserve(capacity);
    for (int tid = 0; tid < capacity; ++tid)
      workers_.emplace_back([this, tid] { worker_loop(tid); });
  }

  void worker_loop(int tid) {
    if (!allowed_cpus_.empty()) {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(allowed_cpus_[static_cast<std::size_t>(tid) % allowed_cpus_.size()], &mask);
      (void)pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    }
    std::uint64_t seen = 0;
    for (;;) {
      std::function<void(int)> task;
      if (e8_spin_) {
        for (int spin = 0; spin < 20000; ++spin) {
          if (generation_.load(std::memory_order_acquire) != seen) break;
          _mm_pause();
        }
      }
      pthread_mutex_lock(&mu_);
      while (!stop_ && generation_.load(std::memory_order_acquire) == seen)
        pthread_cond_wait(&start_cv_, &mu_);
      if (stop_) {
        pthread_mutex_unlock(&mu_);
        return;
      }
      seen = generation_.load(std::memory_order_acquire);
      if (tid >= active_) {
        pthread_mutex_unlock(&mu_);
        continue;
      }
      task = task_;
      pthread_mutex_unlock(&mu_);
      try {
        task(tid);
      } catch (...) {
        pthread_mutex_lock(&mu_);
        if (!failure_) failure_ = std::current_exception();
        pthread_mutex_unlock(&mu_);
      }
      if (e8_atomic_done_) {
        if (completed_atomic_.fetch_add(1, std::memory_order_acq_rel) + 1 == active_) {
          pthread_mutex_lock(&mu_);
          pthread_cond_signal(&done_cv_);
          pthread_mutex_unlock(&mu_);
        }
      } else {
        pthread_mutex_lock(&mu_);
        ++completed_;
        if (completed_ == active_) pthread_cond_signal(&done_cv_);
        pthread_mutex_unlock(&mu_);
      }
    }
  }

  std::mutex submit_mu_;
  pthread_mutex_t mu_;
  pthread_cond_t start_cv_, done_cv_;
  std::vector<std::thread> workers_;
  std::vector<int> allowed_cpus_;
  std::function<void(int)> task_;
  std::exception_ptr failure_;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<int> completed_atomic_{0};
  int active_ = 0, completed_ = 0;
  bool e8_atomic_done_ = false, e8_spin_ = false;
  bool stop_ = false;
};

template <class Fn>
inline void parallel_workers(int threads, Fn&& fn) {
  PersistentWorkerPool::instance(threads).run(threads, std::forward<Fn>(fn));
}

template <class Fn>
inline void parallel_range(std::int64_t count, int threads, Fn&& fn) {
  parallel_workers(threads, [&](int tid) {
    const std::int64_t begin = count * tid / threads;
    const std::int64_t end = count * (tid + 1) / threads;
    if (begin < end) fn(begin, end);
  });
}

int round_up(int x, int q) { return (x + q - 1) / q * q; }

inline bf16 to_bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<bf16>(bits >> 16);
}

inline float from_bf16(bf16 value) {
  std::uint32_t bits=static_cast<std::uint32_t>(value)<<16;
  float out;
  std::memcpy(&out,&bits,sizeof(out));
  return out;
}

template <class T> class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t n)
      : p_(static_cast<T*>(_mm_malloc(n * sizeof(T), 64))), n_(n) {
    TORCH_CHECK(p_ != nullptr || n == 0, "AMX-v2 aligned allocation failed");
  }
  ~AlignedBuffer() { _mm_free(p_); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  T* data() { return p_; }
  const T* data() const { return p_; }
  std::size_t size() const { return n_; }
 private:
  T* p_;
  std::size_t n_;
};

struct Scratch {
  AlignedBuffer<bf16> y;
  AlignedBuffer<bf16> yt;
  AlignedBuffer<bf16> hp;
  Scratch(int rows, int dp, int kp)
      : y(static_cast<std::size_t>(rows) * dp),
        yt(static_cast<std::size_t>(dp) * rows),
        hp(static_cast<std::size_t>(rows) * kp) {}
};

inline __m512 load_bf16x16_fp32(const bf16* p) {
  const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 16));
}

inline __m512 load_bf16_tail_fp32(const bf16* p, int lanes) {
  const __mmask16 mask=static_cast<__mmask16>((1u<<lanes)-1u);
  const __m256i x=_mm256_maskz_loadu_epi16(mask,p);
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(x),16));
}

inline __m512 load_bf16_tail_scalar_fp32(const bf16* p, int lanes) {
  alignas(64) float tmp[16]{};
  for (int i=0;i<lanes;++i) tmp[i]=from_bf16(p[i]);
  return _mm512_load_ps(tmp);
}

inline void store_fp32_tail_bf16(bf16* p, __m512 x, int lanes) {
  const __mmask16 mask=static_cast<__mmask16>((1u<<lanes)-1u);
  const __m256bh packed=_mm512_cvtneps_pbh(x);
  _mm256_mask_storeu_epi16(p,mask,(__m256i)packed);
}

template <class Index>
void pull_panel_impl(const std::int64_t* rp, const Index* ci,
                const bf16* src, int logical_d, int stride,
                int row0, int valid, bf16* out, int out_stride,
                bool vector_bf16_store, bool single_scan_special,
                const std::uint8_t* active_rows, bool single_scan_wide) {
  if (logical_d == 128 && stride == 128) {
    for (int local = 0; local < valid; ++local) {
      const int row = row0 + local;
      bf16* d=out+static_cast<std::size_t>(local)*out_stride;
      const bf16* self = src + static_cast<std::size_t>(row) * stride;
      __m512 a0,a1,a2,a3,a4,a5,a6,a7;
      if (active_rows == nullptr || active_rows[row] != 0) {
        a0=load_bf16x16_fp32(self);a1=load_bf16x16_fp32(self+16);
        a2=load_bf16x16_fp32(self+32);a3=load_bf16x16_fp32(self+48);
        a4=load_bf16x16_fp32(self+64);a5=load_bf16x16_fp32(self+80);
        a6=load_bf16x16_fp32(self+96);a7=load_bf16x16_fp32(self+112);
      } else {
        a0=_mm512_setzero_ps();a1=_mm512_setzero_ps();
        a2=_mm512_setzero_ps();a3=_mm512_setzero_ps();
        a4=_mm512_setzero_ps();a5=_mm512_setzero_ps();
        a6=_mm512_setzero_ps();a7=_mm512_setzero_ps();
      }
      for (std::int64_t e=rp[row]; e<rp[row+1]; ++e) {
        const int source=static_cast<int>(ci[e]);
        if (active_rows != nullptr && active_rows[source] == 0) continue;
        const bf16* p=src+static_cast<std::size_t>(source)*stride;
        a0=_mm512_add_ps(a0,load_bf16x16_fp32(p));a1=_mm512_add_ps(a1,load_bf16x16_fp32(p+16));
        a2=_mm512_add_ps(a2,load_bf16x16_fp32(p+32));a3=_mm512_add_ps(a3,load_bf16x16_fp32(p+48));
        a4=_mm512_add_ps(a4,load_bf16x16_fp32(p+64));a5=_mm512_add_ps(a5,load_bf16x16_fp32(p+80));
        a6=_mm512_add_ps(a6,load_bf16x16_fp32(p+96));a7=_mm512_add_ps(a7,load_bf16x16_fp32(p+112));
      }
      if (vector_bf16_store) {
        _mm512_storeu_si512(reinterpret_cast<void*>(d),
            (__m512i)_mm512_cvtne2ps_pbh(a1,a0));
        _mm512_storeu_si512(reinterpret_cast<void*>(d+32),
            (__m512i)_mm512_cvtne2ps_pbh(a3,a2));
        _mm512_storeu_si512(reinterpret_cast<void*>(d+64),
            (__m512i)_mm512_cvtne2ps_pbh(a5,a4));
        _mm512_storeu_si512(reinterpret_cast<void*>(d+96),
            (__m512i)_mm512_cvtne2ps_pbh(a7,a6));
      } else {
        alignas(64) float v[128];
        _mm512_store_ps(v,a0);_mm512_store_ps(v+16,a1);_mm512_store_ps(v+32,a2);_mm512_store_ps(v+48,a3);
        _mm512_store_ps(v+64,a4);_mm512_store_ps(v+80,a5);_mm512_store_ps(v+96,a6);_mm512_store_ps(v+112,a7);
        for(int q=0;q<128;++q)d[q]=to_bf16(v[q]);
      }
    }
    return;
  }
  // Wide Transform-HighD slab variant.  The legacy generic path below scans
  // the same CSR row once per 16-wide feature block.  For an explicitly gated
  // macro slab, keep one accumulator per output block and traverse each row's
  // CSR list only once.  The accumulation order within every block remains
  // self row followed by CSR neighbors, so BF16 rounding and FP32 sums match
  // the reference path up to normal reduction-order differences.
  if (single_scan_wide && logical_d > 128 && stride >= logical_d) {
    const int blocks = (logical_d + 15) / 16;
    std::vector<__m512> acc(static_cast<std::size_t>(blocks));
    for (int local = 0; local < valid; ++local) {
      const int row = row0 + local;
      std::fill(acc.begin(), acc.end(), _mm512_setzero_ps());
      auto add = [&](int source) {
        if (active_rows != nullptr && active_rows[source] == 0) return;
        const bf16* q = src + static_cast<std::size_t>(source) * stride;
        for (int b = 0; b < blocks; ++b) {
          const int q0 = b * 16;
          const int lanes = std::min(16, logical_d - q0);
          const __m512 v = lanes == 16
              ? load_bf16x16_fp32(q + q0)
              : load_bf16_tail_fp32(q + q0, lanes);
          acc[static_cast<std::size_t>(b)] = _mm512_add_ps(
              acc[static_cast<std::size_t>(b)], v);
        }
      };
      add(row);
      for (std::int64_t e = rp[row]; e < rp[row + 1]; ++e)
        add(static_cast<int>(ci[e]));
      bf16* dst = out + static_cast<std::size_t>(local) * out_stride;
      for (int b = 0; b < blocks; ++b) {
        const int q0 = b * 16;
        const int lanes = std::min(16, logical_d - q0);
        if (lanes == 16 && vector_bf16_store) {
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + q0),
              (__m256i)_mm512_cvtneps_pbh(
                  acc[static_cast<std::size_t>(b)]));
        } else {
          alignas(64) float tmp[16];
          _mm512_store_ps(tmp, acc[static_cast<std::size_t>(b)]);
          for (int q = 0; q < lanes; ++q)
            dst[q0 + q] = to_bf16(tmp[q]);
        }
      }
    }
    return;
  }
  // Generic small-D single-scan path.  The historical implementation only
  // had hand-written D=47/100/128 cases, which made the benchmark's selected
  // kernel depend on the dataset shape.  Keep one CSR traversal per output
  // panel for every 1 <= D <= 128; each output block owns one ZMM accumulator
  // and the final BF16 conversion preserves the per-block accumulation order.
  if (single_scan_special && logical_d <= 128 && stride >= logical_d) {
    const int blocks=(logical_d+15)/16;
    for(int local=0;local<valid;++local){
      const int row=row0+local;
      __m512 acc[8];
      for(int b=0;b<blocks;++b)acc[b]=_mm512_setzero_ps();
      auto add=[&](int source){
        if(active_rows!=nullptr && active_rows[source]==0)return;
        const bf16* q=src+static_cast<std::size_t>(source)*stride;
        for(int b=0;b<blocks;++b){
          const int q0=b*16;
          const int lanes=std::min(16,logical_d-q0);
          const __m512 v=(lanes==16)?load_bf16x16_fp32(q+q0):
              load_bf16_tail_fp32(q+q0,lanes);
          acc[b]=_mm512_add_ps(acc[b],v);
        }
      };
      add(row);
      for(std::int64_t e=rp[row];e<rp[row+1];++e)
        add(static_cast<int>(ci[e]));
      bf16* dst=out+static_cast<std::size_t>(local)*out_stride;
      for(int b=0;b<blocks;++b){
        const int q0=b*16;
        const int lanes=std::min(16,logical_d-q0);
        if(lanes==16)
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+q0),
              (__m256i)_mm512_cvtneps_pbh(acc[b]));
        else
          store_fp32_tail_bf16(dst+q0,acc[b],lanes);
      }
    }
    return;
  }
  if (single_scan_special && logical_d == 100 && stride == 100) {
    for (int local=0;local<valid;++local) {
      const int row=row0+local;
      const bf16* self=src+static_cast<std::size_t>(row)*stride;
      __m512 a0=load_bf16x16_fp32(self),a1=load_bf16x16_fp32(self+16);
      __m512 a2=load_bf16x16_fp32(self+32),a3=load_bf16x16_fp32(self+48);
      __m512 a4=load_bf16x16_fp32(self+64),a5=load_bf16x16_fp32(self+80);
      __m512 a6=load_bf16_tail_scalar_fp32(self+96,4);
      for(std::int64_t e=rp[row];e<rp[row+1];++e){
        const bf16* q=src+static_cast<std::size_t>(ci[e])*stride;
        a0=_mm512_add_ps(a0,load_bf16x16_fp32(q));
        a1=_mm512_add_ps(a1,load_bf16x16_fp32(q+16));
        a2=_mm512_add_ps(a2,load_bf16x16_fp32(q+32));
        a3=_mm512_add_ps(a3,load_bf16x16_fp32(q+48));
        a4=_mm512_add_ps(a4,load_bf16x16_fp32(q+64));
        a5=_mm512_add_ps(a5,load_bf16x16_fp32(q+80));
        a6=_mm512_add_ps(a6,load_bf16_tail_scalar_fp32(q+96,4));
      }
      bf16* dst=out+static_cast<std::size_t>(local)*out_stride;
      _mm512_storeu_si512(reinterpret_cast<void*>(dst),
          (__m512i)_mm512_cvtne2ps_pbh(a1,a0));
      _mm512_storeu_si512(reinterpret_cast<void*>(dst+32),
          (__m512i)_mm512_cvtne2ps_pbh(a3,a2));
      _mm512_storeu_si512(reinterpret_cast<void*>(dst+64),
          (__m512i)_mm512_cvtne2ps_pbh(a5,a4));
      alignas(64) float tail[16];_mm512_store_ps(tail,a6);
      for(int i=0;i<4;++i)dst[96+i]=to_bf16(tail[i]);
    }
    return;
  }
  if (single_scan_special && logical_d == 47 && stride == 47) {
    for (int local=0;local<valid;++local) {
      const int row=row0+local;
      const bf16* self=src+static_cast<std::size_t>(row)*stride;
      __m512 a0=load_bf16x16_fp32(self),a1=load_bf16x16_fp32(self+16);
      __m512 a2=load_bf16_tail_fp32(self+32,15);
      for(std::int64_t e=rp[row];e<rp[row+1];++e){
        const bf16* q=src+static_cast<std::size_t>(ci[e])*stride;
        a0=_mm512_add_ps(a0,load_bf16x16_fp32(q));
        a1=_mm512_add_ps(a1,load_bf16x16_fp32(q+16));
        a2=_mm512_add_ps(a2,load_bf16_tail_fp32(q+32,15));
      }
      bf16* dst=out+static_cast<std::size_t>(local)*out_stride;
      _mm512_storeu_si512(reinterpret_cast<void*>(dst),
          (__m512i)_mm512_cvtne2ps_pbh(a1,a0));
      store_fp32_tail_bf16(dst+32,a2,15);
    }
    return;
  }
  // Backward pads D=47 to a 64-element source/output stride.  The generic
  // path below scans the same CSR three times, once per 16-wide feature
  // block.  Keep the per-feature accumulation order unchanged while scanning
  // each row once.  When active_rows is present, a source is skipped only if
  // every already-quantized BF16 grad_scaled element in that row is zero.
  if (single_scan_special && logical_d == 47 && stride == 64) {
    for (int local=0;local<valid;++local) {
      const int row=row0+local;
      __m512 a0=_mm512_setzero_ps();
      __m512 a1=_mm512_setzero_ps();
      __m512 a2=_mm512_setzero_ps();
      auto add=[&](int source) {
        if(active_rows!=nullptr && active_rows[source]==0) return;
        const bf16* q=src+static_cast<std::size_t>(source)*stride;
        a0=_mm512_add_ps(a0,load_bf16x16_fp32(q));
        a1=_mm512_add_ps(a1,load_bf16x16_fp32(q+16));
        a2=_mm512_add_ps(a2,load_bf16_tail_fp32(q+32,15));
      };
      add(row);
      for(std::int64_t e=rp[row];e<rp[row+1];++e)
        add(static_cast<int>(ci[e]));
      bf16* dst=out+static_cast<std::size_t>(local)*out_stride;
      _mm512_storeu_si512(reinterpret_cast<void*>(dst),
          (__m512i)_mm512_cvtne2ps_pbh(a1,a0));
      store_fp32_tail_bf16(dst+32,a2,15);
    }
    return;
  }
  for (int local=0; local<valid; ++local) {
    const int row=row0+local;
    for(int q=0;q<logical_d;q+=16) {
      const int lanes=std::min(16,logical_d-q);
      __m512 sum=_mm512_setzero_ps();
      auto add=[&](int j){
        if(lanes==16) sum=_mm512_add_ps(sum,load_bf16x16_fp32(src+static_cast<std::size_t>(j)*stride+q));
        else {
          alignas(64) float tmp[16]{};
          const bf16* p=src+static_cast<std::size_t>(j)*stride+q;
          for(int x=0;x<lanes;++x){std::uint32_t b=static_cast<std::uint32_t>(p[x])<<16;std::memcpy(tmp+x,&b,4);}
          sum=_mm512_add_ps(sum,_mm512_load_ps(tmp));
        }
      };
      add(row);for(std::int64_t e=rp[row];e<rp[row+1];++e)add(static_cast<int>(ci[e]));
      alignas(64) float tmp[16];_mm512_store_ps(tmp,sum);
      bf16* d=out+static_cast<std::size_t>(local)*out_stride+q;
      for(int x=0;x<lanes;++x)d[x]=to_bf16(tmp[x]);
    }
  }
}

void pull_panel(const std::int64_t* rp, const std::int64_t* ci64,
                const std::int32_t* ci32, bool use_int32,
                const bf16* src, int logical_d, int stride,
                int row0, int valid, bf16* out, int out_stride,
                bool vector_bf16_store, bool single_scan_special,
                const std::uint8_t* active_rows=nullptr,
                bool single_scan_wide=false) {
  if (use_int32)
    pull_panel_impl(rp,ci32,src,logical_d,stride,row0,valid,out,out_stride,
                    vector_bf16_store,single_scan_special,active_rows,
                    single_scan_wide);
  else
    pull_panel_impl(rp,ci64,src,logical_d,stride,row0,valid,out,out_stride,
                    vector_bf16_store,single_scan_special,active_rows,
                    single_scan_wide);
}

// Fused source-scale pull used by the transform-High-D immediate-consume
// candidate.  The old path first materializes an [N,Dp] BF16 ``Gs`` tensor
// and then scans CSR once per D slab.  This helper applies the source row
// scale, rounds each source value to BF16, and accumulates the rounded value
// directly while traversing the requested row panel.  It deliberately keeps
// the established BF16-round-before-sum semantics instead of accumulating
// FP32 source values, so it is numerically comparable to the materialized
// path.  The helper is opt-in until its full graph gate is complete.
template <class Index>
void pull_panel_scaled_grad_impl(
    const std::int64_t* rp, const Index* ci, const float* grad,
    const float* scale, int logical_d, int grad_stride, int row0, int valid,
    bf16* out, int out_stride) {
  for (int local = 0; local < valid; ++local) {
    const int row = row0 + local;
    bf16* dst = out + static_cast<std::size_t>(local) * out_stride;
    for (int q = 0; q < logical_d; q += 16) {
      const int lanes = std::min(16, logical_d - q);
      __m512 sum = _mm512_setzero_ps();
      alignas(64) bf16 rounded[16]{};
      auto add = [&](int source) {
        const float* src = grad + static_cast<std::size_t>(source) *
            grad_stride + q;
        const __m512 sv = _mm512_set1_ps(scale[source]);
        if (lanes == 16) {
          const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(src), sv);
          const __m256bh packed = _mm512_cvtneps_pbh(v);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(rounded),
                              (__m256i)packed);
          sum = _mm512_add_ps(sum, load_bf16x16_fp32(rounded));
        } else {
          alignas(64) float tmp[16]{};
          for (int x = 0; x < lanes; ++x)
            tmp[x] = from_bf16(to_bf16(src[x] * scale[source]));
          sum = _mm512_add_ps(sum, _mm512_load_ps(tmp));
        }
      };
      add(row);
      for (std::int64_t e = rp[row]; e < rp[row + 1]; ++e)
        add(static_cast<int>(ci[e]));
      if (lanes == 16) {
        store_fp32_tail_bf16(dst + q, sum, 16);
      } else {
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, sum);
        for (int x = 0; x < lanes; ++x)
          dst[q + x] = to_bf16(tmp[x]);
      }
    }
    for (int q = logical_d; q < out_stride; ++q)
      dst[q] = bf16(0);
  }
}

void pull_panel_scaled_grad(
    const std::int64_t* rp, const std::int64_t* ci64,
    const std::int32_t* ci32, bool use_int32, const float* grad,
    const float* scale, int logical_d, int grad_stride, int row0, int valid,
    bf16* out, int out_stride) {
  if (use_int32)
    pull_panel_scaled_grad_impl(rp, ci32, grad, scale, logical_d, grad_stride,
                                row0, valid, out, out_stride);
  else
    pull_panel_scaled_grad_impl(rp, ci64, grad, scale, logical_d, grad_stride,
                                row0, valid, out, out_stride);
}

struct Int32IndexWorkspace {
  const std::int64_t* identity;
  std::int64_t count;
  AlignedBuffer<std::int32_t> values;
  Int32IndexWorkspace(const std::int64_t* ci,std::int64_t count_)
      : identity(ci),count(count_),values(static_cast<std::size_t>(count_)) {
    for(std::int64_t e=0;e<count;++e){
      TORCH_CHECK(ci[e]>=0 && ci[e]<=INT32_MAX,
                  "E9 int32 colidx conversion out of range");
      values.data()[e]=static_cast<std::int32_t>(ci[e]);
    }
  }
};

const std::int32_t* int32_colidx_workspace(const std::int64_t* ci,
                                           std::int64_t count,bool& reused) {
  static std::mutex mu;
  static std::vector<std::unique_ptr<Int32IndexWorkspace>> cache;
  std::lock_guard<std::mutex> lock(mu);
  for(auto& ws:cache)if(ws->identity==ci && ws->count==count){
    reused=true;
    return ws->values.data();
  }
  reused=false;
  cache.emplace_back(std::make_unique<Int32IndexWorkspace>(ci,count));
  return cache.back()->values.data();
}

std::vector<std::vector<int>> panel_schedule(const std::int64_t* rp, int n,
                                              int panel, int threads) {
  const int count=(n+panel-1)/panel;
  std::vector<std::pair<std::uint64_t,int>> work; work.reserve(count);
  for(int p=0;p<count;++p){int a=p*panel,b=std::min(n,a+panel);work.push_back({static_cast<std::uint64_t>(rp[b]-rp[a]),p});}
  std::sort(work.begin(),work.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
  std::vector<std::uint64_t> load(threads);std::vector<std::vector<int>> own(threads);
  for(auto [nnz,p]:work){int t=static_cast<int>(std::min_element(load.begin(),load.end())-load.begin());own[t].push_back(p);load[t]+=nnz;}
  for(auto& x:own)std::sort(x.begin(),x.end());return own;
}

struct SourceSignature {
  std::uint64_t bits[4]{};
};

inline std::uint64_t source_hash(std::uint64_t x) {
  // SplitMix64 gives stable, cheap hashes without a per-panel allocation.
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline void source_signature_add(SourceSignature& sig, std::uint64_t source) {
  const std::uint64_t block=source >> 8;  // 256 source rows per locality bin.
  const std::uint64_t h1=source_hash(block) & 255ULL;
  const std::uint64_t h2=source_hash(block ^ 0xd6e8feb86659fd93ULL) & 255ULL;
  sig.bits[h1 >> 6] |= 1ULL << (h1 & 63);
  sig.bits[h2 >> 6] |= 1ULL << (h2 & 63);
}

inline int source_signature_overlap(const SourceSignature& a,
                                    const SourceSignature& b) {
  return __builtin_popcountll(a.bits[0]&b.bits[0]) +
         __builtin_popcountll(a.bits[1]&b.bits[1]) +
         __builtin_popcountll(a.bits[2]&b.bits[2]) +
         __builtin_popcountll(a.bits[3]&b.bits[3]);
}

inline int source_signature_fresh(const SourceSignature& sig,
                                  const SourceSignature& hot) {
  return __builtin_popcountll(sig.bits[0]&~hot.bits[0]) +
         __builtin_popcountll(sig.bits[1]&~hot.bits[1]) +
         __builtin_popcountll(sig.bits[2]&~hot.bits[2]) +
         __builtin_popcountll(sig.bits[3]&~hot.bits[3]);
}

inline void source_signature_merge(SourceSignature& hot,
                                   const SourceSignature& sig) {
  for(int i=0;i<4;++i)hot.bits[i] |= sig.bits[i];
}

struct SourcePanel {
  std::uint64_t nnz=0;
  std::uint64_t source_key=UINT64_MAX;
  int panel=0;
  SourceSignature sig;
};

// NUMA-P4 scheduler from the plan: assign NNZ-heavy panels to a NUMA state
// using a small source-block Bloom signature, then greedily choose the least
// loaded worker inside that NUMA.  The signature is only a scheduling hint;
// CSR traversal and accumulation order inside every panel are untouched.
std::vector<std::vector<int>> panel_schedule_source_reuse(
    const std::int64_t* rp, const std::int64_t* ci, int n, int panel,
    int threads) {
  const int count=(n+panel-1)/panel;
  std::vector<SourcePanel> work;
  work.reserve(count);
  for(int p=0;p<count;++p){
    const int a=p*panel,b=std::min(n,a+panel);
    SourcePanel item;
    item.panel=p;
    item.nnz=static_cast<std::uint64_t>(rp[b]-rp[a]);
    for(std::int64_t e=rp[a];e<rp[b];++e){
      const std::uint64_t source=static_cast<std::uint64_t>(ci[e]);
      item.source_key=std::min(item.source_key,source);
      source_signature_add(item.sig,source);
    }
    work.emplace_back(item);
  }
  std::sort(work.begin(),work.end(),[](const SourcePanel& a,const SourcePanel& b){
    if(a.nnz!=b.nnz)return a.nnz>b.nnz;
    return a.panel<b.panel;
  });
  const int per_numa=runtime_per_numa(threads);
  const int numas=std::max(1,(threads+per_numa-1)/per_numa);
  std::vector<SourceSignature> hot(static_cast<std::size_t>(numas));
  std::vector<SourceSignature> thread_hot(static_cast<std::size_t>(threads));
  std::vector<std::uint64_t> numa_load(static_cast<std::size_t>(numas),0);
  std::vector<std::uint64_t> thread_load(static_cast<std::size_t>(threads),0);
  std::vector<std::vector<int>> own(static_cast<std::size_t>(threads));
  std::vector<std::uint64_t> source_keys(static_cast<std::size_t>(count),UINT64_MAX);
  std::uint64_t total=0;
  for(const auto& item:work){total+=item.nnz;source_keys[item.panel]=item.source_key;}
  const double target=static_cast<double>(std::max<std::uint64_t>(1,total)) /
                      static_cast<double>(numas);
  const double thread_target=static_cast<double>(std::max<std::uint64_t>(1,total)) /
                             static_cast<double>(threads);
  for(const auto& item:work){
    int best_z=0;
    double best_score=std::numeric_limits<double>::infinity();
    for(int z=0;z<numas;++z){
      const double load=(static_cast<double>(numa_load[z])+item.nnz)/target;
      const double fresh=static_cast<double>(source_signature_fresh(item.sig,hot[z]))/256.0;
      const double overlap=static_cast<double>(source_signature_overlap(item.sig,hot[z]))/256.0;
      const double score=load+0.05*fresh-0.05*overlap;
      if(score<best_score){best_score=score;best_z=z;}
    }
    const int first=best_z*per_numa,last=std::min(threads,(best_z+1)*per_numa);
    int best_t=first;
    double best_thread_score=std::numeric_limits<double>::infinity();
    for(int t=first;t<last;++t){
      const double load=(static_cast<double>(thread_load[t])+item.nnz)/thread_target;
      const double fresh=static_cast<double>(source_signature_fresh(
          item.sig,thread_hot[t]))/256.0;
      const double overlap=static_cast<double>(source_signature_overlap(
          item.sig,thread_hot[t]))/256.0;
      const double score=load+0.05*fresh-0.05*overlap;
      if(score<best_thread_score){best_thread_score=score;best_t=t;}
    }
    own[best_t].push_back(item.panel);
    thread_load[best_t]+=item.nnz;
    numa_load[best_z]+=item.nnz;
    source_signature_merge(hot[best_z],item.sig);
    source_signature_merge(thread_hot[best_t],item.sig);
  }
  for(auto& panels:own)
    std::sort(panels.begin(),panels.end(),[&](int a,int b){
      if(source_keys[a]!=source_keys[b])return source_keys[a]<source_keys[b];
      return a<b;
    });
  return own;
}

void reduce_dwt(const float* local, int threads, int per_numa, int elems,
                float* out) {
  const int group=std::max(1,std::min(4,per_numa));
  const int groups=(threads+group-1)/group, numas=(threads+per_numa-1)/per_numa;
  std::vector<float> gv(static_cast<std::size_t>(groups)*elems),nv(static_cast<std::size_t>(numas)*elems);
  for(int g=0;g<groups;++g)for(int t=g*group;t<std::min(threads,(g+1)*group);++t)
    for(int i=0;i<elems;++i)gv[static_cast<std::size_t>(g)*elems+i]+=local[static_cast<std::size_t>(t)*elems+i];
  for(int z=0;z<numas;++z){int a=z*per_numa,b=std::min(threads,(z+1)*per_numa);
    for(int g=a/group;g<(b+group-1)/group;++g)for(int i=0;i<elems;++i)nv[static_cast<std::size_t>(z)*elems+i]+=gv[static_cast<std::size_t>(g)*elems+i];}
  std::fill(out,out+elems,0.0f);for(int z=0;z<numas;++z)for(int i=0;i<elems;++i)out[i]+=nv[static_cast<std::size_t>(z)*elems+i];
}

enum class BackwardImplKind : std::uint8_t {
  standard_c3 = 0,
  aggregate_static_v3 = 1,
};

struct BackwardWorkspaceSpec {
  int n, logical_d, logical_k, dp, kp, threads, panel;
  const std::int64_t* rp;
  const std::int64_t* ci;
  bool locality;
  bool direct_hs;
  bool compute_dx;
  bool need_active;
  bool need_db_local;
  bool need_schedule;
  BackwardImplKind impl;
};

inline std::size_t workspace_cache_limit_bytes() {
  const char* value = std::getenv("TFS_WORKSPACE_CACHE_MAX_BYTES");
  if (value == nullptr || *value == '\0')
    return static_cast<std::size_t>(512) << 20;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  TORCH_CHECK(end != value && *end == '\0' && parsed > 0,
              "TFS_WORKSPACE_CACHE_MAX_BYTES must be a positive integer");
  return static_cast<std::size_t>(parsed);
}

struct BackwardWorkspace {
  int n, logical_d, logical_k, dp, kp, threads, panel;
  bool locality;
  bool direct_hs;
  bool compute_dx;
  bool need_active;
  bool need_db_local;
  bool need_schedule;
  BackwardImplKind impl;
  const std::int64_t* rp_identity;
  const std::int64_t* ci_identity;
  AlignedBuffer<bf16> gs, hb, wt;
  AlignedBuffer<std::uint8_t> active;
  AlignedBuffer<float> local, dwt, group_reduce, numa_reduce, db_local;
  AlignedBuffer<bf16> packed_wt;
  std::vector<std::unique_ptr<Scratch>> scratch;
  std::vector<std::vector<int>> own;
  std::size_t bytes_total = 0;
  std::uint64_t last_used = 0;
  bool last_cache_hit = false;

  explicit BackwardWorkspace(const BackwardWorkspaceSpec& spec)
      : n(spec.n), logical_d(spec.logical_d), logical_k(spec.logical_k), dp(spec.dp), kp(spec.kp), threads(spec.threads),
        panel(spec.panel), locality(spec.locality),
        direct_hs(spec.direct_hs), compute_dx(spec.compute_dx),
        need_active(spec.need_active), need_db_local(spec.need_db_local),
        need_schedule(spec.need_schedule), impl(spec.impl),
        rp_identity(spec.rp), ci_identity(spec.ci),
        gs(static_cast<std::size_t>(n) * dp),
        hb(direct_hs ? 0 : static_cast<std::size_t>(n) * kp),
        wt(compute_dx ? static_cast<std::size_t>(dp) * kp : 0),
        active(need_active ? static_cast<std::size_t>(n) : 0),
        local(static_cast<std::size_t>(threads) * dp * kp),
        dwt(static_cast<std::size_t>(dp) * kp),
        group_reduce(static_cast<std::size_t>(
            (threads + runtime_reduce_group(threads) - 1) /
            runtime_reduce_group(threads)) * dp * kp),
        numa_reduce(static_cast<std::size_t>((threads + runtime_per_numa(threads) - 1) /
                                             runtime_per_numa(threads)) * dp * kp),
        db_local(need_db_local ? static_cast<std::size_t>(threads) * dp : 0),
        packed_wt(compute_dx ?
                  static_cast<std::size_t>(dp / 32) * (kp / 16) * 512 : 0),
        own(need_schedule ?
            (locality ? panel_schedule_source_reuse(
                spec.rp,spec.ci,n,panel,threads)
                      : panel_schedule(spec.rp,n,panel,threads))
            : std::vector<std::vector<int>>(
                static_cast<std::size_t>(threads))) {
    scratch.reserve(threads);
    for(int t=0;t<threads;++t) {
      auto z=std::make_unique<Scratch>(panel,dp,kp);
      scratch.emplace_back(std::move(z));
    }
    bytes_total=(local.size()+dwt.size()+
        group_reduce.size()+numa_reduce.size()+db_local.size())*sizeof(float) +
        (gs.size()+hb.size()+wt.size())*sizeof(bf16) +
        packed_wt.size()*sizeof(bf16) + active.size()*sizeof(std::uint8_t) +
        scratch.size()*(static_cast<std::size_t>(panel)*dp*2 +
                        static_cast<std::size_t>(panel)*kp)*sizeof(bf16);
    bool first_touch=numa_first_touch_enabled(threads);
    const char* ft_mode=std::getenv("TFS_NUMA_FIRST_TOUCH");
    const bool ft_auto=(ft_mode==nullptr || *ft_mode=='\0' ||
                        std::strcmp(ft_mode,"auto")==0);
    // Do not pay a worker-pool/page-fault setup cost for tiny probes; the
    // automatic gate is intended for real multi-megabyte training workspaces.
    if (ft_auto && bytes_total < (4u<<20)) first_touch=false;
    auto fill_bf16=[](AlignedBuffer<bf16>& buffer) {
      if (buffer.size())
        std::fill(buffer.data(),buffer.data()+buffer.size(),bf16(0));
    };
    auto fill_float=[](AlignedBuffer<float>& buffer) {
      if (buffer.size())
        std::fill(buffer.data(),buffer.data()+buffer.size(),0.0f);
    };
    if (!first_touch) {
      fill_bf16(gs); fill_bf16(hb); fill_bf16(wt); fill_bf16(packed_wt);
      fill_float(local); fill_float(dwt); fill_float(group_reduce);
      fill_float(numa_reduce); fill_float(db_local);
      if(active.size())
        std::fill(active.data(),active.data()+active.size(),std::uint8_t(0));
      for(auto& z:scratch) {
        std::fill(z->y.data(),z->y.data()+z->y.size(),bf16(0));
        std::fill(z->yt.data(),z->yt.data()+z->yt.size(),bf16(0));
        std::fill(z->hp.data(),z->hp.data()+z->hp.size(),bf16(0));
      }
    } else {
      // First-touch every page from the same persistent workers that will
      // consume it.  This gives Linux a real NUMA placement signal without
      // introducing a libnuma dependency or changing tensor ownership.
      parallel_workers(threads,[&](int tid){
        auto zero_bf16=[&](bf16* p,std::size_t total){
          if(total==0)return;
          const std::size_t b=total*static_cast<std::size_t>(tid)/threads;
          const std::size_t e=total*static_cast<std::size_t>(tid+1)/threads;
          std::fill(p+b,p+e,bf16(0));
        };
        auto zero_float=[&](float* p,std::size_t total){
          if(total==0)return;
          const std::size_t b=total*static_cast<std::size_t>(tid)/threads;
          const std::size_t e=total*static_cast<std::size_t>(tid+1)/threads;
          std::fill(p+b,p+e,0.0f);
        };
        zero_bf16(gs.data(),gs.size());
        zero_bf16(hb.data(),hb.size());
        zero_bf16(wt.data(),wt.size());
        zero_bf16(packed_wt.data(),packed_wt.size());
        zero_float(dwt.data(),dwt.size());
        zero_float(group_reduce.data(),group_reduce.size());
        zero_float(numa_reduce.data(),numa_reduce.size());
        zero_float(db_local.data(),db_local.size());
        if(active.size()){
          const std::size_t b=active.size()*static_cast<std::size_t>(tid)/threads;
          const std::size_t e=active.size()*static_cast<std::size_t>(tid+1)/threads;
          std::fill(active.data()+b,active.data()+e,std::uint8_t(0));
        }
        const std::size_t lb=static_cast<std::size_t>(tid)*dp*kp;
        std::fill(local.data()+lb,local.data()+lb+static_cast<std::size_t>(dp)*kp,0.0f);
        auto& z=*scratch[tid];
        std::fill(z.y.data(),z.y.data()+z.y.size(),bf16(0));
        std::fill(z.yt.data(),z.yt.data()+z.yt.size(),bf16(0));
        std::fill(z.hp.data(),z.hp.data()+z.hp.size(),bf16(0));
      });
    }
    if (internal_profile_enabled())
      std::cout<<"TFS_NUMA first_touch="<<(first_touch?1:0)
               <<",threads="<<threads<<",per_numa="<<runtime_per_numa(threads)
               <<",workspace_bytes="<<bytes_total
               <<",hb_bytes="<<(hb.size()*sizeof(bf16))
               <<",wt_bytes="<<(wt.size()*sizeof(bf16))
               <<",packed_wt_bytes="<<(packed_wt.size()*sizeof(bf16))
               <<",scratch_bytes="<<(scratch.size()*
                   (static_cast<std::size_t>(panel)*dp*2 +
                    static_cast<std::size_t>(panel)*kp)*sizeof(bf16))
               <<std::endl;
  }

  bool matches(const BackwardWorkspaceSpec& spec) const {
    return n==spec.n && logical_d==spec.logical_d && logical_k==spec.logical_k && dp==spec.dp && kp==spec.kp &&
           threads==spec.threads && panel==spec.panel &&
           locality==spec.locality && direct_hs==spec.direct_hs &&
           compute_dx==spec.compute_dx && need_active==spec.need_active &&
           need_db_local==spec.need_db_local &&
           need_schedule==spec.need_schedule && impl==spec.impl &&
           rp_identity==spec.rp && ci_identity==spec.ci;
  }
};

std::mutex& backward_workspace_mutex() {
  static std::mutex mu;
  return mu;
}

BackwardWorkspace& backward_workspace(const BackwardWorkspaceSpec& request,
                                      bool& reused, bool& transient_allocation) {
  static std::vector<std::unique_ptr<BackwardWorkspace>> cache;
  static std::unique_ptr<BackwardWorkspace> transient;
  static std::size_t cache_bytes=0;
  static std::uint64_t use_clock=0;
  transient.reset();
  BackwardWorkspaceSpec spec=request;
  spec.locality=spec.need_schedule &&
      locality_schedule_enabled(spec.n,spec.threads);
  for(auto& ws:cache) if(ws->matches(spec)) {
    reused=true; transient_allocation=false;
    ws->last_used=++use_clock;
    ws->last_cache_hit=true;
    return *ws;
  }
  reused=false; transient_allocation=false;
  auto candidate=std::make_unique<BackwardWorkspace>(spec);
  candidate->last_used=++use_clock;
  candidate->last_cache_hit=false;
  const std::size_t limit=workspace_cache_limit_bytes();
  if(candidate->bytes_total>limit){
    transient_allocation=true; transient=std::move(candidate);
    return *transient;
  }
  while(!cache.empty() && cache_bytes+candidate->bytes_total>limit){
    auto victim=std::min_element(
        cache.begin(),cache.end(),[](const auto& lhs,const auto& rhs){
          return lhs->last_used<rhs->last_used;
        });
    cache_bytes-=(*victim)->bytes_total;
    cache.erase(victim);
  }
  cache_bytes+=candidate->bytes_total;
  cache.emplace_back(std::move(candidate));
  return *cache.back();
}

void pack_rhs_into(const bf16* b,int reduction_padded,int output_padded,bf16* packed) {
  const int output_blocks=output_padded/16;
  for(int k=0;k<reduction_padded;++k) for(int n=0;n<output_padded;++n) {
    const std::size_t base=(static_cast<std::size_t>(k/32)*output_blocks+n/16)*512;
    packed[base+static_cast<std::size_t>((k%32)/2)*32+2*(n%16)+(k&1)]=
        b[static_cast<std::size_t>(k)*output_padded+n];
  }
}

void reduce_dwt_persistent(const float* local,int threads,int per_numa,int elems,
                           float* gv,float* nv,float* out) {
  const int group=std::max(1,std::min(4,per_numa));
  const int groups=(threads+group-1)/group,numas=(threads+per_numa-1)/per_numa;
  std::fill(gv,gv+static_cast<std::size_t>(groups)*elems,0.0f);
  std::fill(nv,nv+static_cast<std::size_t>(numas)*elems,0.0f);
  for(int g=0;g<groups;++g) for(int t=g*group;t<std::min(threads,(g+1)*group);++t)
    for(int i=0;i<elems;++i) gv[static_cast<std::size_t>(g)*elems+i]+=local[static_cast<std::size_t>(t)*elems+i];
  for(int z=0;z<numas;++z){int a=z*per_numa,b=std::min(threads,(z+1)*per_numa);
    for(int g=a/group;g<(b+group-1)/group;++g)for(int i=0;i<elems;++i)nv[static_cast<std::size_t>(z)*elems+i]+=gv[static_cast<std::size_t>(g)*elems+i];}
  std::fill(out,out+elems,0.0f);
  for(int z=0;z<numas;++z)for(int i=0;i<elems;++i)out[i]+=nv[static_cast<std::size_t>(z)*elems+i];
}

void reduce_dwt_parallel_deterministic(const float* local,int threads,int per_numa,
                                       int elems,float* out) {
  const int group=std::max(1,std::min(4,per_numa));
  const int groups=(threads+group-1)/group;
  const int numas=(threads+per_numa-1)/per_numa;
  parallel_workers(threads,[&](int tid){
    const int begin=elems*tid/threads,end=elems*(tid+1)/threads;
    for(int i=begin;i<end;++i){
      float group_sums[32]{};
      float numa_sums[8]{};
      for(int g=0;g<groups;++g)
        for(int t=g*group;t<std::min(threads,(g+1)*group);++t)
          group_sums[g]+=local[static_cast<std::size_t>(t)*elems+i];
      for(int z=0;z<numas;++z){
        const int a=z*per_numa,b=std::min(threads,(z+1)*per_numa);
        for(int g=a/group;g<(b+group-1)/group;++g)numa_sums[z]+=group_sums[g];
      }
      float sum=0.0f;
      for(int z=0;z<numas;++z)sum+=numa_sums[z];
      out[i]=sum;
    }
  });
}

struct ForwardScheduleWorkspace {
  int n,threads,panel;
  bool locality;
  const std::int64_t* rp_identity;
  const std::int64_t* ci_identity;
  std::vector<std::vector<int>> own;
  ForwardScheduleWorkspace(int n_,int threads_,int panel_,const std::int64_t* rp,
                           const std::int64_t* ci,bool locality_)
      : n(n_),threads(threads_),panel(panel_),locality(locality_),
        rp_identity(rp),ci_identity(ci),
        own(locality_ ? panel_schedule_source_reuse(rp,ci,n,panel,threads)
                      : panel_schedule(rp,n,panel,threads)) {}
  bool matches(int n_,int threads_,int panel_,const std::int64_t* rp,
               const std::int64_t* ci,bool locality_) const {
    return n==n_ && threads==threads_ && panel==panel_ &&
           locality==locality_ && rp_identity==rp && ci_identity==ci;
  }
};

const std::vector<std::vector<int>>& forward_schedule_workspace(
    int n,int threads,int panel,const std::int64_t* rp,const std::int64_t* ci,
    bool& reused) {
  static std::mutex mu;
  static std::vector<std::unique_ptr<ForwardScheduleWorkspace>> cache;
  const bool locality=locality_schedule_enabled(n,threads);
  std::lock_guard<std::mutex> lock(mu);
  for(auto& ws:cache)if(ws->matches(n,threads,panel,rp,ci,locality)){
    reused=true;
    return ws->own;
  }
  reused=false;
  cache.emplace_back(std::make_unique<ForwardScheduleWorkspace>(n,threads,panel,rp,ci,locality));
  return cache.back()->own;
}

// Build the exact BF16 first-layer representation consumed by the AMX C3
// paths.  This is deliberately kept as a small, side-effect-free producer so
// r5 can materialize it once and pass the resulting tensor back to the
// existing consumer.  The conversion order and the optional E5 vector path
// are identical to the historical in-forward producer.
at::Tensor build_hs_bf16_exact(const at::Tensor& x, const at::Tensor& s,
                               int threads, double* alloc_end = nullptr,
                               double* fill_end = nullptr) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat &&
              s.device().is_cpu() && s.scalar_type() == at::kFloat,
              "Hs producer expects CPU FP32 x and scale");
  TORCH_CHECK(x.dim() == 2 && s.dim() == 1 && s.numel() == x.size(0) &&
              threads >= 1 && threads <= 32,
              "Hs producer shape/thread contract failed");
  const int n = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const float* xp = x.data_ptr<float>();
  const float* sp = s.data_ptr<float>();
  auto hs = at::empty({n, k}, x.options().dtype(at::kBFloat16));
  const double allocation_done = now_ms();
  if (alloc_end != nullptr) *alloc_end = allocation_done;
  bf16* hsp = reinterpret_cast<bf16*>(hs.data_ptr<at::BFloat16>());
  const bool glue_e5 = experiment_flag("TFS_GLUE_E5_VEC_HS");
  if (glue_e5) {
    parallel_range(n, threads, [&](std::int64_t begin, std::int64_t end) {
      for (std::int64_t i = begin; i < end; ++i) {
        const __m512 sv = _mm512_set1_ps(sp[i]);
        const float* xr = xp + static_cast<std::size_t>(i) * k;
        bf16* hr = hsp + static_cast<std::size_t>(i) * k;
        int q = 0;
        for (; q + 32 <= k; q += 32) {
          const __m512 v0 = _mm512_mul_ps(_mm512_loadu_ps(xr + q), sv);
          const __m512 v1 = _mm512_mul_ps(_mm512_loadu_ps(xr + q + 16), sv);
          _mm512_storeu_si512(reinterpret_cast<void*>(hr + q),
                              (__m512i)_mm512_cvtne2ps_pbh(v1, v0));
        }
        for (; q < k; ++q) hr[q] = to_bf16(xr[q] * sp[i]);
      }
    });
  } else {
    parallel_range(n, threads, [&](std::int64_t begin, std::int64_t end) {
      for (std::int64_t i = begin; i < end; ++i)
        for (int q = 0; q < k; ++q)
          hsp[static_cast<std::size_t>(i) * k + q] =
              to_bf16(xp[static_cast<std::size_t>(i) * k + q] * sp[i]);
    });
  }
  if (fill_end != nullptr) *fill_end = now_ms();
  return hs;
}

// V2 keeps the logical feature width unchanged but stores Hs at an AMX-safe
// physical row stride.  Only the tail columns are zero-filled; the logical
// columns use the exact same FP32 multiply/BF16 conversion as the V1 producer.
at::Tensor build_hs_bf16_padded_exact(const at::Tensor& x, const at::Tensor& s,
                                      int threads, int physical_k) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat &&
              s.device().is_cpu() && s.scalar_type() == at::kFloat,
              "padded Hs producer expects CPU FP32 x and scale");
  TORCH_CHECK(x.dim() == 2 && s.dim() == 1 && s.numel() == x.size(0) &&
              threads >= 1 && threads <= 32,
              "padded Hs producer shape/thread contract failed");
  const int n = static_cast<int>(x.size(0));
  const int logical_k = static_cast<int>(x.size(1));
  TORCH_CHECK(physical_k >= logical_k && (physical_k % 64) == 0,
              "padded Hs physical width must be >= logical K and 64-aligned");
  const float* xp = x.data_ptr<float>();
  const float* sp = s.data_ptr<float>();
  auto hs = at::empty({n, physical_k},
                      x.options().dtype(at::kBFloat16));
  bf16* hsp = reinterpret_cast<bf16*>(hs.data_ptr<at::BFloat16>());
  const bool glue_e5 = experiment_flag("TFS_GLUE_E5_VEC_HS");
  parallel_range(n, threads, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; ++i) {
      const __m512 sv = _mm512_set1_ps(sp[i]);
      const float* xr = xp + static_cast<std::size_t>(i) * logical_k;
      bf16* hr = hsp + static_cast<std::size_t>(i) * physical_k;
      int q = 0;
      if (glue_e5) {
        for (; q + 32 <= logical_k; q += 32) {
          const __m512 v0 = _mm512_mul_ps(_mm512_loadu_ps(xr + q), sv);
          const __m512 v1 = _mm512_mul_ps(_mm512_loadu_ps(xr + q + 16), sv);
          _mm512_storeu_si512(reinterpret_cast<void*>(hr + q),
                              (__m512i)_mm512_cvtne2ps_pbh(v1, v0));
        }
      }
      for (; q < logical_k; ++q)
        hr[q] = to_bf16(xr[q] * sp[i]);
      for (; q < physical_k; ++q) hr[q] = bf16(0);
    }
  });
  return hs;
}

} // namespace

static std::vector<at::Tensor> c3_forward_amx_impl(
    const at::Tensor& x_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, bool transform_first, bool return_saved_t,
    bool allow_wide_aggregate = false,
    const at::Tensor* cached_hs_in = nullptr,
    bool padded_cached_hs = false,
    const at::Tensor* cached_hs_replicas_in = nullptr) {
  const double profile_t0=now_ms();
  auto x=x_in.contiguous(),w=weight_in.contiguous(),b=bias_in.contiguous();
  auto rp_t=rowptr_in.contiguous(),ci_t=colidx_in.contiguous(),s=scale_in.contiguous();
  TORCH_CHECK(x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat && b.scalar_type()==at::kFloat && s.scalar_type()==at::kFloat,"AMX-v2 forward expects FP32 tensors");
  TORCH_CHECK(rp_t.scalar_type()==at::kLong && ci_t.scalar_type()==at::kLong,"AMX-v2 forward CSR must be int64");
  const int n=static_cast<int>(x.size(0)),k=static_cast<int>(x.size(1)),d=static_cast<int>(w.size(1)),threads=static_cast<int>(threads64),panel=512;
  const bool glue_e2=experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bool glue_e4=experiment_flag("TFS_GLUE_E4_FUSED_EPILOGUE");
  const bool glue_e5=experiment_flag("TFS_GLUE_E5_VEC_HS");
  const bool glue_e7=experiment_flag("TFS_GLUE_E7_FORWARD_SCHEDULE");
  const bool glue_e9=experiment_flag("TFS_GLUE_E9_INT32_COLIDX");
  const bool fwd_v2=experiment_flag("TFS_FWD_V2_SINGLE_SCAN");
  // Wide-K probe: the existing pull/packing paths are stride-parametric; keep
  // the optimized AMX output contract (D<=128) while allowing K>128.
  // The stock C3 contract keeps D<=128.  Aggregate-first wide output is a
  // separate path: the sparse pull remains K-wide (K<=128 in the caller),
  // while the dense BF16 GEMM/epilogue handles the large class dimension.
  TORCH_CHECK(w.size(0)==k && b.numel()==d && s.numel()==n && k>=1 &&
              (d<=128 || (allow_wide_aggregate && !transform_first && d>128)),
              "AMX-v2 wide-K forward shape unsupported");
  const float* xp=x.data_ptr<float>();const float* sp=s.data_ptr<float>();
  const std::int64_t* rp=rp_t.data_ptr<std::int64_t>();const std::int64_t* ci=ci_t.data_ptr<std::int64_t>();
  bool e9_index_reused=false;
  const bool use_e9=formal_colidx_enabled(glue_e9,ci,ci_t.numel());
  const std::int32_t* ci32=use_e9?
      int32_colidx_workspace(ci,ci_t.numel(),e9_index_reused):nullptr;
  const double profile_setup=now_ms();
  const bool hs_replicated = cached_hs_replicas_in != nullptr;
  const bool hs_cached = cached_hs_in != nullptr || hs_replicated;
  TORCH_CHECK(!(cached_hs_in != nullptr && hs_replicated),
              "cached Hs and replicated Hs cannot be supplied together");
  at::Tensor hs;
  int hs_stride = k;
  int hs_replica_count = 1;
  const bf16* hs_replica_base = nullptr;
  double profile_hs_alloc = profile_setup;
  double profile_hs = profile_setup;
  if (hs_replicated) {
    const at::Tensor& replicas = *cached_hs_replicas_in;
    TORCH_CHECK(replicas.device().is_cpu() &&
                replicas.scalar_type() == at::kBFloat16 &&
                replicas.dim() == 3 && replicas.is_contiguous() &&
                replicas.size(0) >= 1 && replicas.size(1) == n &&
                (replicas.size(2) == k ||
                 (padded_cached_hs &&
                  replicas.size(2) == round_up(k, 64))),
                "replicated Hs must be contiguous CPU BF16 [R,N,Kp]");
    hs = replicas.select(0, 0);
    hs_stride = static_cast<int>(replicas.size(2));
    hs_replica_count = static_cast<int>(replicas.size(0));
    hs_replica_base = reinterpret_cast<const bf16*>(
        replicas.data_ptr<at::BFloat16>());
  } else if (hs_cached) {
    TORCH_CHECK(cached_hs_in->device().is_cpu() &&
                cached_hs_in->scalar_type() == at::kBFloat16 &&
                cached_hs_in->dim() == 2 && cached_hs_in->is_contiguous() &&
                cached_hs_in->size(0) == n &&
                (cached_hs_in->size(1) == k ||
                 (padded_cached_hs &&
                  cached_hs_in->size(1) == round_up(k, 64))),
                "cached Hs must be contiguous CPU BF16 with logical/padded x shape");
    hs = *cached_hs_in;
    hs_stride = static_cast<int>(hs.size(1));
  } else {
    hs = build_hs_bf16_exact(x, s, threads, &profile_hs_alloc,
                             &profile_hs);
  }
  bf16* hsp = reinterpret_cast<bf16*>(hs.data_ptr<at::BFloat16>());
  auto wb=w.to(at::kBFloat16);
  const double profile_w=now_ms();
  at::Tensor aggregated;
  // High-D aggregate GEMM probe.  The established path uses the framework
  // BF16 matmul and then a vectorized FP32 epilogue.  The dimension-driven
  // gate below can instead write the FP32 accumulator directly and apply
  // scale+bias in the same output buffer; an environment override is kept for
  // paired ablations against oneDNN/ATen.
  at::Tensor amx_aggregate_out;
  bool used_amx_aggregate_gemm = false;
  bool used_amx_aggregate_gemm4 = false;
  at::Tensor saved_t;
  double schedule_ms=0.0,sparse_ms=0.0,dense_ms=0.0,intermediate_alloc_ms=0.0;
  const bool locality_schedule=locality_schedule_enabled(n,threads);
  std::vector<int> worker_replica_slots;
  if (hs_replicated) {
    worker_replica_slots.resize(static_cast<std::size_t>(threads));
    for (int tid = 0; tid < threads; ++tid) {
      worker_replica_slots[static_cast<std::size_t>(tid)] =
          runtime_worker_numa_slot(tid, threads) %
          std::max(1, hs_replica_count);
    }
  }
  if(transform_first){
    const double q0=now_ms();
    auto hs_logical = hs.narrow(1, 0, k);
    auto transformed=at::matmul(hs_logical,wb).contiguous();
    dense_ms=now_ms()-q0;
    const double q1=now_ms();
    aggregated=at::empty({n,d},transformed.options());
    const bf16* src=reinterpret_cast<const bf16*>(transformed.data_ptr<at::BFloat16>());bf16* dst=reinterpret_cast<bf16*>(aggregated.data_ptr<at::BFloat16>());
    intermediate_alloc_ms=now_ms()-q1;
    const double q2=now_ms();
    std::vector<std::vector<int>> own_local;
    bool schedule_reused=false;
    const std::vector<std::vector<int>>* own_ptr=nullptr;
    if(glue_e7)own_ptr=&forward_schedule_workspace(n,threads,panel,rp,ci,schedule_reused);
    else {own_local=locality_schedule?
                panel_schedule_source_reuse(rp,ci,n,panel,threads):
                panel_schedule(rp,n,panel,threads);own_ptr=&own_local;}
    const auto& own=*own_ptr;
    schedule_ms=now_ms()-q2;
    const double q3=now_ms();
    parallel_workers(threads,[&](int tid){
      for(int p:own[tid]){
        int r0=p*panel,v=std::min(panel,n-r0);
        pull_panel(rp,ci,ci32,use_e9,src,d,d,r0,v,
                   dst+static_cast<std::size_t>(r0)*d,d,glue_e2,fwd_v2);
      }
    });
    sparse_ms=now_ms()-q3;
  }else{
    const double q0=now_ms();
    auto pulled=at::empty({n,k},hs.options());bf16* dst=reinterpret_cast<bf16*>(pulled.data_ptr<at::BFloat16>());
    if (return_saved_t) saved_t=pulled;
    intermediate_alloc_ms=now_ms()-q0;
    const double q1=now_ms();
    std::vector<std::vector<int>> own_local;
    bool schedule_reused=false;
    const std::vector<std::vector<int>>* own_ptr=nullptr;
    if(glue_e7)own_ptr=&forward_schedule_workspace(n,threads,panel,rp,ci,schedule_reused);
    else {own_local=locality_schedule?
                panel_schedule_source_reuse(rp,ci,n,panel,threads):
                panel_schedule(rp,n,panel,threads);own_ptr=&own_local;}
    const auto& own=*own_ptr;
    schedule_ms=now_ms()-q1;
    const double q2=now_ms();
    parallel_workers(threads,[&](int tid){
      for(int p:own[tid]){
        int r0=p*panel,v=std::min(panel,n-r0);
        const bf16* worker_hsp = hsp;
        if (hs_replicated) {
          const int replica = worker_replica_slots[
              static_cast<std::size_t>(tid)];
          worker_hsp = hs_replica_base +
              static_cast<std::size_t>(replica) * n * hs_stride;
        }
        pull_panel(rp,ci,ci32,use_e9,worker_hsp,k,hs_stride,r0,v,
                   dst+static_cast<std::size_t>(r0)*k,k,glue_e2,fwd_v2);
      }
    });
    sparse_ms=now_ms()-q2;
    const double q3=now_ms();
    // Dimension/work-size planner default: wide-output aggregate layers with
    // enough rows amortize weight packing and benefit from the fused FP32
    // epilogue.  The environment variable remains an explicit override for
    // paired gates; no dataset name or graph-specific branch is involved.
    const bool amx_gemm_auto = d >= 512 && n >= 4096;
    const bool amx_gemm_base = allow_wide_aggregate &&
        highd_adaptive_flag("TFS_HIGHD_AMX_GEMM", amx_gemm_auto) &&
        d > 128;
    const bool amx_gemm4 = amx_gemm_base &&
        highd_adaptive_flag("TFS_HIGHD_AMX_GEMM4", true);
    // The four-tile kernel zero-pads a K/N tail in a 16-row scratch block;
    // the older one-tile primitive still requires complete rows and a
    // 32-aligned reduction.  Keep that explicit control arm safe rather than
    // silently reading past a logical tensor.
    const bool amx_gemm_gate = amx_gemm_base &&
        (amx_gemm4 || ((k % 32) == 0 && (n % 16) == 0));
    if (amx_gemm_gate) {
      const int kp = round_up(k, 32);
      const int dp = round_up(d, 16);
      const bf16* wbp = reinterpret_cast<const bf16*>(
          wb.data_ptr<at::BFloat16>());
      std::vector<bf16> wpad(static_cast<std::size_t>(kp) * dp, bf16(0));
      for (int q = 0; q < k; ++q)
        std::memcpy(wpad.data() + static_cast<std::size_t>(q) * dp,
                    wbp + static_cast<std::size_t>(q) * d,
                    static_cast<std::size_t>(d) * sizeof(bf16));
      std::vector<bf16> packed_w(
          static_cast<std::size_t>(kp / 32) * (dp / 16) * 512, bf16(0));
      pack_rhs_into(wpad.data(), kp, dp, packed_w.data());
      amx_aggregate_out = at::empty({n, dp}, x.options());
      const bf16* ap = reinterpret_cast<const bf16*>(
          pulled.data_ptr<at::BFloat16>());
      float* cp = amx_aggregate_out.data_ptr<float>();
      const float* scale_p = s.data_ptr<float>();
      const float* bias_p = b.data_ptr<float>();
      parallel_workers(threads, [&](int tid) {
        bv2::configure_amx_tiles_16x64();
        std::vector<bf16> a_pad;
        if (kp != k || (n % 16) != 0)
          a_pad.resize(static_cast<std::size_t>(16) * kp);
        const int row_tiles = (n + 15) / 16;
        const int tile_begin = row_tiles * tid / threads;
        const int tile_end = row_tiles * (tid + 1) / threads;
        const int row_begin = tile_begin * 16;
        const int row_end = tile_end * 16;
        for (int row = row_begin; row < row_end; row += 16) {
          const int valid = std::min(16, n - row);
          const bf16* a_block = ap + static_cast<std::size_t>(row) * k;
          if (valid != 16 || kp != k) {
            std::fill(a_pad.begin(), a_pad.end(), bf16(0));
            for (int i = 0; i < valid; ++i)
              std::memcpy(a_pad.data() + static_cast<std::size_t>(i) * kp,
                          ap + static_cast<std::size_t>(row + i) * k,
                          static_cast<std::size_t>(k) * sizeof(bf16));
            a_block = a_pad.data();
          }
          if (amx_gemm4) {
            bv2::amx_gemm_4c_epilogue(
                a_block, 16, kp, packed_w, dp, row, valid, d, scale_p, bias_p,
                cp + static_cast<std::size_t>(row) * dp);
          } else {
            bv2::amx_gemm_1c_baseline(
                a_block, 16, kp, packed_w,
                dp, cp + static_cast<std::size_t>(row) * dp);
          }
        }
        _tile_release();
      });
      used_amx_aggregate_gemm = true;
      used_amx_aggregate_gemm4 = amx_gemm4;
    } else {
      aggregated=at::matmul(pulled,wb);
    }
    dense_ms=now_ms()-q3;
  }
  const double profile_ep0=now_ms();
  at::Tensor out;
  if (used_amx_aggregate_gemm && used_amx_aggregate_gemm4) {
    out = amx_aggregate_out.narrow(1, 0, d);
  } else if (used_amx_aggregate_gemm) {
    // AMX produced FP32 accumulators directly.  Apply the same affine
    // epilogue as the BF16 framework path, but avoid materialising a second
    // FP32 tensor or converting an intermediate BF16 matrix back to FP32.
    const int dp = static_cast<int>(amx_aggregate_out.size(1));
    out = amx_aggregate_out.narrow(1, 0, d);
    float* outp = amx_aggregate_out.data_ptr<float>();
    const float* bp = b.data_ptr<float>();
    parallel_range(n, threads, [&](std::int64_t begin, std::int64_t end) {
      for (std::int64_t i = begin; i < end; ++i) {
        const __m512 sv = _mm512_set1_ps(sp[i]);
        float* orow = outp + static_cast<std::size_t>(i) * dp;
        int q = 0;
        for (; q + 16 <= d; q += 16) {
          _mm512_storeu_ps(
              orow + q,
              _mm512_fmadd_ps(_mm512_loadu_ps(orow + q), sv,
                              _mm512_loadu_ps(bp + q)));
        }
        for (; q < d; ++q) orow[q] = orow[q] * sp[i] + bp[q];
      }
    });
  } else if (glue_e4) {
    out=at::empty({n,d},x.options());
    const bf16* aggp=reinterpret_cast<const bf16*>(aggregated.data_ptr<at::BFloat16>());
    float* outp=out.data_ptr<float>();
    const float* bp=b.data_ptr<float>();
    parallel_range(n,threads,[&](std::int64_t begin,std::int64_t end){
      for(std::int64_t i=begin;i<end;++i){
        const __m512 sv=_mm512_set1_ps(sp[i]);
        const bf16* ar=aggp+static_cast<std::size_t>(i)*d;
        float* orow=outp+static_cast<std::size_t>(i)*d;
        int q=0;
        for(;q+16<=d;q+=16){
          const __m512 v=load_bf16x16_fp32(ar+q);
          _mm512_storeu_ps(orow+q,_mm512_fmadd_ps(v,sv,_mm512_loadu_ps(bp+q)));
        }
        for(;q<d;++q)orow[q]=from_bf16(ar[q])*sp[i]+bp[q];
      }
    });
  } else {
    out=aggregated.to(at::kFloat);
    out.mul_(s.unsqueeze(1));out.add_(b);
  }
  const double profile_end=now_ms();
  if(internal_profile_enabled()) {
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_INTERNAL,kind=forward,k="<<k<<",d="<<d<<",threads="<<threads
      <<",order="<<(transform_first?"transform":"aggregate")
      <<",glue_e2="<<(glue_e2?1:0)
      <<",glue_e4="<<(glue_e4?1:0)
      <<",glue_e5="<<(glue_e5?1:0)
      <<",glue_e7="<<(glue_e7?1:0)
      <<",glue_e9="<<(glue_e9?1:0)
      <<",fwd_v2_single_scan="<<(fwd_v2?1:0)
      <<",locality_schedule="<<(locality_schedule?1:0)
       <<",e9_index_reused="<<(e9_index_reused?1:0)
       <<",hs_cache_hit="<<(hs_cached?1:0)
      <<",hs_replica_count="<<hs_replica_count
      <<",setup_ms="<<(profile_setup-profile_t0)
      <<",hs_alloc_ms="<<(profile_hs_alloc-profile_setup)
      <<",hs_scale_ms="<<(profile_hs-profile_hs_alloc)
      <<",w_bf16_ms="<<(profile_w-profile_hs)
      <<",intermediate_alloc_ms="<<intermediate_alloc_ms
      <<",schedule_ms="<<schedule_ms<<",sparse_ms="<<sparse_ms
      <<",dense_ms="<<dense_ms<<",amx_aggregate_gemm="
      <<(used_amx_aggregate_gemm?1:0)
      <<",amx_aggregate_gemm4="<<(used_amx_aggregate_gemm4?1:0)
      <<",epilogue_ms="<<(profile_end-profile_ep0)
      <<",total_ms="<<(profile_end-profile_t0)<<std::endl;
  }
  if (return_saved_t) {
    TORCH_CHECK(!transform_first && saved_t.defined(),
                "saved-T forward is only valid for aggregate-first");
    return {out,hs,saved_t};
  }
  return {out,hs};
}

std::vector<at::Tensor> c3_forward_amx_v2(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads, bool transform_first) {
  return c3_forward_amx_impl(x,weight,bias,rowptr,colidx,scale,threads,
                              transform_first,false,false);
}

// Generic aggregate-saved producer.  It returns the normal output, the
// transient Hs buffer (kept for API symmetry/debugging), and the BF16 pulled
// matrix P=B*Hs that the v4 backward consumes.  Unlike the static T0 cache,
// this is rebuilt for every forward and is therefore valid when x requires a
// gradient or changes between steps.
std::vector<at::Tensor> c3_forward_aggregate_saved_amx_v4(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads) {
  return c3_forward_amx_impl(x,weight,bias,rowptr,colidx,scale,threads,
                              false,true,false);
}

// Public r5 producer.  The returned tensor is a canonical contiguous CPU
// BF16 buffer and is safe to retain for repeated static-graph forwards.
at::Tensor c3_prepare_static_hs_v1(const at::Tensor& x_in,
                                   const at::Tensor& scale_in,
                                   int64_t threads64) {
  auto x = x_in.contiguous();
  auto s = scale_in.contiguous();
  const int threads = static_cast<int>(threads64);
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat &&
              s.device().is_cpu() && s.scalar_type() == at::kFloat &&
              x.dim() == 2 && s.dim() == 1 && s.numel() == x.size(0) &&
              threads >= 1 && threads <= 32,
              "static Hs preparation contract failed");
  return build_hs_bf16_exact(x, s, threads);
}

// V2 producer for a persistent Hs buffer with a 64-element AMX-safe row
// stride.  The logical feature dimension remains x.size(1); the tail is zero.
at::Tensor c3_prepare_static_hs_padded_v2(const at::Tensor& x_in,
                                          const at::Tensor& scale_in,
                                          int64_t physical_k64,
                                          int64_t threads64) {
  auto x = x_in.contiguous();
  auto s = scale_in.contiguous();
  const int threads = static_cast<int>(threads64);
  const int physical_k = static_cast<int>(physical_k64);
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat &&
              s.device().is_cpu() && s.scalar_type() == at::kFloat &&
              x.dim() == 2 && s.dim() == 1 && s.numel() == x.size(0) &&
              threads >= 1 && threads <= 32,
              "padded static Hs preparation contract failed");
  return build_hs_bf16_padded_exact(x, s, threads, physical_k);
}

// Build one page-local Hs copy per runtime NUMA ownership group.  The values
// are copied byte-for-byte from the canonical BF16 cache; only the physical
// page placement changes because each worker group first-touches its own
// replica.  This is intentionally an opt-in producer: the extra R*N*K bytes
// are worthwhile only when a large static cache is repeatedly read by workers
// spanning multiple NUMA domains.
at::Tensor c3_replicate_static_hs_numa_v1(const at::Tensor& hs_in,
                                          int64_t threads64) {
  const int threads = static_cast<int>(threads64);
  TORCH_CHECK(hs_in.device().is_cpu() &&
              hs_in.scalar_type() == at::kBFloat16 &&
              hs_in.dim() == 2 && hs_in.is_contiguous() &&
              hs_in.size(0) >= 1 && hs_in.size(1) >= 1 &&
              threads >= 1 && threads <= 32,
              "NUMA Hs replica producer expects contiguous CPU BF16 [N,K]");
  const int n = static_cast<int>(hs_in.size(0));
  const int k = static_cast<int>(hs_in.size(1));
  const int per_numa = runtime_per_numa(threads);
  const int replicas = hs_replica_count_for_threads(threads);
  auto out = at::empty({replicas, n, k}, hs_in.options());
  const bf16* src = reinterpret_cast<const bf16*>(
      hs_in.data_ptr<at::BFloat16>());
  bf16* dst = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  const std::size_t row_bytes = static_cast<std::size_t>(k) * sizeof(bf16);
  parallel_workers(threads, [&](int tid) {
    const int ownership_slot = runtime_worker_numa_slot(tid, threads);
    const int replica = ownership_slot % replicas;
    int group_threads = 0;
    int local_tid = 0;
    for (int other = 0; other < threads; ++other) {
      if (runtime_worker_numa_slot(other, threads) % replicas == replica) {
        ++group_threads;
        if (other < tid) ++local_tid;
      }
    }
    group_threads = std::max(1, group_threads);
    const int row_begin = n * local_tid / group_threads;
    const int row_end = n * (local_tid + 1) / group_threads;
    bf16* replica_dst = dst +
        static_cast<std::size_t>(replica) * n * k;
    for (int row = row_begin; row < row_end; ++row) {
      std::memcpy(replica_dst + static_cast<std::size_t>(row) * k,
                  src + static_cast<std::size_t>(row) * k, row_bytes);
    }
  });
  if (internal_profile_enabled()) {
    std::cout << std::fixed << std::setprecision(6)
              << "TFS_INTERNAL,kind=hs_replica_build,threads=" << threads
              << ",replicas=" << replicas << ",per_numa=" << per_numa
              << ",n=" << n << ",k=" << k << ",bytes="
              << (out.numel() * out.element_size()) << std::endl;
  }
  return out;
}

// Optional V3 producer for an aggregate-first static layer.  It materializes
// T0 = B * Q_BF16(S * X) once, using the same CSR pull, BF16 conversion and
// schedule flags as the ordinary aggregate-first C3 forward.  The API is
// deliberately opt-in: transform-first layers cannot use this cache because
// their dense product depends on the changing weight matrix.
at::Tensor c3_scale_grad_bf16_v1(
    const at::Tensor&, const at::Tensor&, int64_t);
std::vector<at::Tensor> c3_backward_saved_t_amx_v3(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, int64_t);

// Forward contract for the generic aggregate-saved path.  Unlike the
// historical static T0 producer this returns the per-forward pulled BF16
// tensor, so it is valid even when the layer input requires dX.
std::vector<at::Tensor> c3_forward_aggregate_saved_amx_v4(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t);

at::Tensor c3_prepare_static_aggregate_v3(
    const at::Tensor& x_in, const at::Tensor& scale_in,
    const at::Tensor& rowptr_in, const at::Tensor& colidx_in,
    int64_t threads64) {
  auto x=x_in.contiguous(), s=scale_in.contiguous();
  auto rp_t=rowptr_in.contiguous(), ci_t=colidx_in.contiguous();
  const int threads=static_cast<int>(threads64);
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat &&
              s.device().is_cpu() && s.scalar_type()==at::kFloat &&
              rp_t.device().is_cpu() && rp_t.scalar_type()==at::kLong &&
              ci_t.device().is_cpu() && ci_t.scalar_type()==at::kLong &&
              x.dim()==2 && s.dim()==1 && s.numel()==x.size(0) &&
              rp_t.dim()==1 && rp_t.numel()==x.size(0)+1 &&
              ci_t.dim()==1 && threads>=1 && threads<=32,
              "static aggregate cache producer contract failed");
  const int n=static_cast<int>(x.size(0));
  const int k=static_cast<int>(x.size(1));
  TORCH_CHECK(k>=1 && k<=128,
              "static aggregate cache currently requires 1<=K<=128");
  auto hs=build_hs_bf16_exact(x,s,threads);
  auto pulled=at::empty({n,k},hs.options());
  const std::int64_t* rp=rp_t.data_ptr<std::int64_t>();
  const std::int64_t* ci=ci_t.data_ptr<std::int64_t>();
  const bool glue_e2=experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bool glue_e7=experiment_flag("TFS_GLUE_E7_FORWARD_SCHEDULE");
  const bool glue_e9=experiment_flag("TFS_GLUE_E9_INT32_COLIDX");
  const bool fwd_v2=experiment_flag("TFS_FWD_V2_SINGLE_SCAN");
  bool e9_index_reused=false;
  const bool use_e9=formal_colidx_enabled(glue_e9,ci,ci_t.numel());
  const std::int32_t* ci32=use_e9?
      int32_colidx_workspace(ci,ci_t.numel(),e9_index_reused):nullptr;
  const int panel=512;
  std::vector<std::vector<int>> own_local;
  bool schedule_reused=false;
  const bool locality_schedule=locality_schedule_enabled(n,threads);
  const std::vector<std::vector<int>>* own_ptr=nullptr;
  if(glue_e7) own_ptr=&forward_schedule_workspace(
      n,threads,panel,rp,ci,schedule_reused);
  else { own_local=locality_schedule?
              panel_schedule_source_reuse(rp,ci,n,panel,threads):
              panel_schedule(rp,n,panel,threads); own_ptr=&own_local; }
  const auto& own=*own_ptr;
  const bf16* src=reinterpret_cast<const bf16*>(
      hs.data_ptr<at::BFloat16>());
  bf16* dst=reinterpret_cast<bf16*>(pulled.data_ptr<at::BFloat16>());
  parallel_workers(threads,[&](int tid){
    for(int p:own[tid]){
      const int row0=p*panel;
      const int valid=std::min(panel,n-row0);
      pull_panel(rp,ci,ci32,use_e9,src,k,k,row0,valid,
                 dst+static_cast<std::size_t>(row0)*k,k,
                 glue_e2,fwd_v2);
    }
  });
  if(internal_profile_enabled()) {
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_INTERNAL,kind=aggregate_cache_build,k="<<k
      <<",threads="<<threads<<",e9_index_reused="
      <<(e9_index_reused?1:0)<<",schedule_reused="
      <<(schedule_reused?1:0)<<",bytes="
      <<(pulled.numel()*pulled.element_size())<<std::endl;
  }
  return pulled;
}

// V3 aggregate-first consumer.  The static T0 cache removes only the sparse
// pull; the changing weight, destination scale, bias and BF16/FP32 epilogue
// remain on the established forward path.
std::vector<at::Tensor> c3_forward_cached_aggregate_amx_v3(
    const at::Tensor& x_in, const at::Tensor& pulled_in,
    const at::Tensor& weight_in, const at::Tensor& bias_in,
    const at::Tensor& rowptr_in, const at::Tensor& colidx_in,
    const at::Tensor& scale_in, int64_t threads64) {
  const double profile_t0=now_ms();
  auto x=x_in.contiguous(), pulled=pulled_in.contiguous();
  auto w=weight_in.contiguous(), b=bias_in.contiguous();
  auto rp=rowptr_in.contiguous(), ci=colidx_in.contiguous();
  auto s=scale_in.contiguous();
  const int n=static_cast<int>(x.size(0));
  const int k=static_cast<int>(x.size(1));
  const int d=static_cast<int>(w.size(1));
  const int threads=static_cast<int>(threads64);
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat &&
              pulled.device().is_cpu() && pulled.scalar_type()==at::kBFloat16 &&
              pulled.is_contiguous() && pulled.dim()==2 &&
              pulled.size(0)==n && pulled.size(1)==k &&
              w.device().is_cpu() && w.scalar_type()==at::kFloat &&
              w.dim()==2 && w.size(0)==k && d>=1 && d<=128 &&
              b.device().is_cpu() && b.scalar_type()==at::kFloat &&
              b.numel()==d && s.device().is_cpu() &&
              s.scalar_type()==at::kFloat && s.numel()==n &&
              rp.scalar_type()==at::kLong && ci.scalar_type()==at::kLong &&
              threads>=1 && threads<=32,
              "cached aggregate consumer contract failed");
  auto wb=w.to(at::kBFloat16);
  const double profile_dense0=now_ms();
  auto aggregated=at::matmul(pulled,wb);
  const double profile_dense1=now_ms();
  at::Tensor out;
  const bool glue_e4=experiment_flag("TFS_GLUE_E4_FUSED_EPILOGUE");
  if(glue_e4) {
    out=at::empty({n,d},x.options());
    const bf16* ap=reinterpret_cast<const bf16*>(
        aggregated.data_ptr<at::BFloat16>());
    float* op=out.data_ptr<float>();
    const float* sp=s.data_ptr<float>();
    const float* bp=b.data_ptr<float>();
    parallel_range(n,threads,[&](std::int64_t begin,std::int64_t end){
      for(std::int64_t i=begin;i<end;++i){
        const __m512 sv=_mm512_set1_ps(sp[i]);
        const bf16* ar=ap+static_cast<std::size_t>(i)*d;
        float* orow=op+static_cast<std::size_t>(i)*d;
        int q=0;
        for(;q+16<=d;q+=16){
          const __m512 v=load_bf16x16_fp32(ar+q);
          _mm512_storeu_ps(orow+q,_mm512_fmadd_ps(
              v,sv,_mm512_loadu_ps(bp+q)));
        }
        for(;q<d;++q) orow[q]=from_bf16(ar[q])*sp[i]+bp[q];
      }
    });
  } else {
    out=aggregated.to(at::kFloat);
    out.mul_(s.unsqueeze(1));
    out.add_(b);
  }
  const double profile_end=now_ms();
  if(internal_profile_enabled()) {
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_INTERNAL,kind=forward_aggregate_cached,k="<<k
      <<",d="<<d<<",threads="<<threads<<",sparse_ms=0.000000"
      <<",dense_ms="<<(profile_dense1-profile_dense0)
      <<",epilogue_ms="<<(profile_end-profile_dense1)
      <<",total_ms="<<(profile_end-profile_t0)<<std::endl;
  }
  return {out,pulled};
}

// V3 backward for a static aggregate cache.  It is valid only when the
// layer input does not require dX (the layer-0 full-batch training case).
// dW uses dW=T0^T * Q_BF16(S*G), while db stays the existing FP32 reduction.
std::vector<at::Tensor> c3_backward_cached_aggregate_amx_v3(
    const at::Tensor& grad_in, const at::Tensor& pulled_in,
    const at::Tensor& rowptr_in, const at::Tensor& colidx_in,
    const at::Tensor& scale_in, int64_t threads64) {
  auto grad=grad_in.contiguous(), pulled=pulled_in.contiguous();
  auto rowptr=rowptr_in.contiguous(), colidx=colidx_in.contiguous();
  auto scale=scale_in.contiguous();
  const int n=static_cast<int>(grad.size(0));
  const int k=static_cast<int>(pulled.size(1));
  TORCH_CHECK(grad.device().is_cpu() && grad.scalar_type()==at::kFloat &&
              pulled.device().is_cpu() && pulled.scalar_type()==at::kBFloat16 &&
              pulled.dim()==2 && pulled.size(0)==n &&
              rowptr.device().is_cpu() && rowptr.scalar_type()==at::kLong &&
              rowptr.numel()==n+1 && colidx.device().is_cpu() &&
              colidx.scalar_type()==at::kLong && scale.device().is_cpu() &&
              scale.scalar_type()==at::kFloat && scale.numel()==n &&
              k>=1 && k<=128,
              "cached aggregate backward contract failed");
  // Reuse the established AMX saved-T dW kernel.  It performs the same
  // BF16 grad scaling, T2 transpose, direct H-panel packing and hierarchical
  // reduction as the normal backward, but intentionally omits the Y=B*Gs
  // sparse pull because T0 already contains B*Hs0.
  return c3_backward_saved_t_amx_v3(
      grad, pulled, rowptr, colidx, scale, threads64);
}

// Consumer entry point for r5.  All downstream sparse, dense and epilogue
// code remains the established C3 implementation; only the Hs producer is
// bypassed.  Requiring contiguous inputs here makes hidden framework/native
// copies observable and keeps the cache gate honest.
std::vector<at::Tensor> c3_forward_cached_hs_amx_v1(
    const at::Tensor& x, const at::Tensor& hs,
    const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads, bool transform_first) {
  TORCH_CHECK(x.is_contiguous() && hs.is_contiguous() && weight.is_contiguous() &&
              bias.is_contiguous() && rowptr.is_contiguous() &&
              colidx.is_contiguous() && scale.is_contiguous(),
              "cached Hs consumer requires contiguous inputs");
  if (hs.dim() == 3) {
    return c3_forward_amx_impl(x, weight, bias, rowptr, colidx, scale, threads,
                               transform_first, false, false, nullptr, false,
                               &hs);
  }
  return c3_forward_amx_impl(x, weight, bias, rowptr, colidx, scale, threads,
                             transform_first, false, false, &hs);
}

std::vector<at::Tensor> c3_forward_cached_hs_padded_amx_v2(
    const at::Tensor& x, const at::Tensor& hs,
    const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads, bool transform_first) {
  TORCH_CHECK(x.is_contiguous() && hs.is_contiguous() &&
              weight.is_contiguous() && bias.is_contiguous() &&
              rowptr.is_contiguous() && colidx.is_contiguous() &&
              scale.is_contiguous(),
              "padded cached Hs consumer requires contiguous inputs");
  TORCH_CHECK((hs.dim() == 2 && hs.size(1) == round_up(x.size(1), 64)) ||
              (hs.dim() == 3 && hs.size(2) == round_up(x.size(1), 64)),
              "padded cached Hs width must be round_up(logical K,64)");
  if (hs.dim() == 3) {
    return c3_forward_amx_impl(x, weight, bias, rowptr, colidx, scale, threads,
                               transform_first, false, false, nullptr, true,
                               &hs);
  }
  return c3_forward_amx_impl(x, weight, bias, rowptr, colidx, scale, threads,
                             transform_first, false, false, &hs, true);
}

// DGL-compatible order for a wide final layer: sparse aggregation in the
// input width, then one full dense GEMM to the class dimension.  Return the
// pre-output-scale pulled tensor as the third result so a custom backward can
// form dW without repeating the sparse aggregation.
std::vector<at::Tensor> c3_forward_aggregate_wide_amx_v3(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads) {
  return c3_forward_amx_impl(x,weight,bias,rowptr,colidx,scale,threads,
                             false,true,true);
}

std::vector<at::Tensor> c3_forward_aggregate_wide_cached_hs_amx_v1(
    const at::Tensor& x, const at::Tensor& hs,
    const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads) {
  TORCH_CHECK(x.is_contiguous() && hs.is_contiguous() &&
              weight.is_contiguous() && bias.is_contiguous() &&
              rowptr.is_contiguous() && colidx.is_contiguous() &&
              scale.is_contiguous(),
              "cached aggregate-wide Hs consumer requires contiguous inputs");
  if (hs.dim() == 3) {
    return c3_forward_amx_impl(x, weight, bias, rowptr, colidx, scale, threads,
                               false, true, true, nullptr, false, &hs);
  }
  return c3_forward_amx_impl(x, weight, bias, rowptr, colidx, scale, threads,
                             false, true, true, &hs);
}

// Sparse-only pull used by the aggregate-first wide backward.  The previous
// implementation called c3_forward_amx_v2 with an identity weight merely to
// obtain A^T(dP).  That path allocated Hs, converted an identity matrix to
// BF16, ran an identity GEMM, and applied a general epilogue.  This primitive
// preserves the same BF16 conversion and CSR accumulation order, but performs
// only the required BF16 sparse pull and final BF16->FP32 conversion.
at::Tensor c3_pull_only_amx_v1(
    const at::Tensor& x_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, int64_t threads64) {
  const double profile_t0=now_ms();
  auto x=x_in.contiguous(),rp_t=rowptr_in.contiguous(),ci_t=colidx_in.contiguous();
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat,
              "pull-only input must be CPU FP32");
  TORCH_CHECK(rp_t.scalar_type()==at::kLong && ci_t.scalar_type()==at::kLong,
              "pull-only CSR must be int64");
  TORCH_CHECK(x.dim()==2 && rp_t.dim()==1 && ci_t.dim()==1,
              "pull-only tensors must have valid dimensions");
  const int n=static_cast<int>(x.size(0));
  const int k=static_cast<int>(x.size(1));
  const int threads=static_cast<int>(threads64),panel=512;
  // The pull loop is feature-stride parametric and handles arbitrary K in
  // 16-wide chunks (with a scalar tail).  The old K<=128 gate only reflected
  // the identity-weight workaround used by the first wide-D prototype; it
  // made a legitimate 256/1024-wide aggregate backward fall back or fail.
  TORCH_CHECK(k>=1 && rp_t.numel()==n+1 && threads>=1 && threads<=32,
              "pull-only AMX supports K>=1 and 1..32 threads");
  const bool glue_e2=experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bool glue_e5=experiment_flag("TFS_GLUE_E5_VEC_HS");
  const bool glue_e7=experiment_flag("TFS_GLUE_E7_FORWARD_SCHEDULE");
  const bool glue_e9=experiment_flag("TFS_GLUE_E9_INT32_COLIDX");
  const bool fwd_v2=experiment_flag("TFS_FWD_V2_SINGLE_SCAN");
  const float* xp=x.data_ptr<float>();
  const std::int64_t* rp=rp_t.data_ptr<std::int64_t>();
  const std::int64_t* ci=ci_t.data_ptr<std::int64_t>();
  bool e9_index_reused=false;
  const bool use_e9=formal_colidx_enabled(glue_e9,ci,ci_t.numel());
  const std::int32_t* ci32=use_e9?
      int32_colidx_workspace(ci,ci_t.numel(),e9_index_reused):nullptr;

  // Match c3_forward_amx_impl's scale=ones Hs construction exactly: convert
  // the FP32 dP rows to BF16 before the sparse accumulation.
  auto hs=at::empty({n,k},x.options().dtype(at::kBFloat16));
  bf16* hsp=reinterpret_cast<bf16*>(hs.data_ptr<at::BFloat16>());
  if (glue_e5) {
    parallel_range(n,threads,[&](std::int64_t begin,std::int64_t end){
      for(std::int64_t i=begin;i<end;++i){
        const float* xr=xp+static_cast<std::size_t>(i)*k;
        bf16* hr=hsp+static_cast<std::size_t>(i)*k;
        int q=0;
        for(;q+32<=k;q+=32){
          const __m512 v0=_mm512_loadu_ps(xr+q);
          const __m512 v1=_mm512_loadu_ps(xr+q+16);
          _mm512_storeu_si512(reinterpret_cast<void*>(hr+q),
              (__m512i)_mm512_cvtne2ps_pbh(v1,v0));
        }
        for(;q<k;++q)hr[q]=to_bf16(xr[q]);
      }
    });
  } else {
    parallel_range(n,threads,[&](std::int64_t begin,std::int64_t end){
      for(std::int64_t i=begin;i<end;++i)
        for(int q=0;q<k;++q)
          hsp[static_cast<std::size_t>(i)*k+q]=
              to_bf16(xp[static_cast<std::size_t>(i)*k+q]);
    });
  }

  auto pulled=at::empty({n,k},hs.options());
  bf16* dst=reinterpret_cast<bf16*>(pulled.data_ptr<at::BFloat16>());
  std::vector<std::vector<int>> own_local;
  bool schedule_reused=false;
  const bool locality_schedule=locality_schedule_enabled(n,threads);
  const std::vector<std::vector<int>>* own_ptr=nullptr;
  if(glue_e7)own_ptr=&forward_schedule_workspace(n,threads,panel,rp,ci,
                                                  schedule_reused);
  else {own_local=locality_schedule?
              panel_schedule_source_reuse(rp,ci,n,panel,threads):
              panel_schedule(rp,n,panel,threads);own_ptr=&own_local;}
  const auto& own=*own_ptr;
  parallel_workers(threads,[&](int tid){
    for(int p:own[tid]){
      const int r0=p*panel,v=std::min(panel,n-r0);
      pull_panel(rp,ci,ci32,use_e9,hsp,k,k,r0,v,
                 dst+static_cast<std::size_t>(r0)*k,k,
                 glue_e2,fwd_v2);
    }
  });

  auto out=at::empty({n,k},x.options());
  const bf16* pp=reinterpret_cast<const bf16*>(pulled.data_ptr<at::BFloat16>());
  float* op=out.data_ptr<float>();
  parallel_range(n,threads,[&](std::int64_t begin,std::int64_t end){
    for(std::int64_t i=begin;i<end;++i){
      const bf16* sr=pp+static_cast<std::size_t>(i)*k;
      float* dr=op+static_cast<std::size_t>(i)*k;
      int q=0;
      for(;q+16<=k;q+=16)
        _mm512_storeu_ps(dr+q,load_bf16x16_fp32(sr+q));
      for(;q<k;++q)dr[q]=from_bf16(sr[q]);
    }
  });
  if(internal_profile_enabled())
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_INTERNAL,kind=pull_only,k="<<k<<",threads="<<threads
      <<",glue_e2="<<(glue_e2?1:0)<<",glue_e5="<<(glue_e5?1:0)
      <<",glue_e7="<<(glue_e7?1:0)<<",glue_e9="<<(glue_e9?1:0)
      <<",e9_index_reused="<<(e9_index_reused?1:0)
      <<",schedule_reused="<<(schedule_reused?1:0)
      <<",total_ms="<<(now_ms()-profile_t0)<<std::endl;
  return out;
}

// BF16-input sibling of c3_pull_only_amx_v1.  Transform-HighD already
// materializes the scaled gradient with the required BF16 rounding.  Feeding
// that tensor directly into the CSR pull avoids the old BF16 -> FP32 -> BF16
// round trip while retaining the exact same pull_panel accumulation order.
at::Tensor c3_pull_only_bf16_amx_v1(
    const at::Tensor& x_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, int64_t threads64) {
  const double profile_t0 = now_ms();
  auto x = x_in.contiguous();
  auto rp_t = rowptr_in.contiguous();
  auto ci_t = colidx_in.contiguous();
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16,
              "BF16 pull-only input must be CPU BF16");
  TORCH_CHECK(rp_t.scalar_type() == at::kLong &&
              ci_t.scalar_type() == at::kLong,
              "BF16 pull-only CSR must be int64");
  TORCH_CHECK(x.dim() == 2 && rp_t.dim() == 1 && ci_t.dim() == 1,
              "BF16 pull-only tensors must have valid dimensions");
  const int n = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int threads = static_cast<int>(threads64);
  const int panel = 512;
  TORCH_CHECK(k >= 1 && rp_t.numel() == n + 1 && threads >= 1 &&
              threads <= 32,
              "BF16 pull-only AMX supports K>=1 and 1..32 threads");

  const bool glue_e2 = experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bool glue_e7 = experiment_flag("TFS_GLUE_E7_FORWARD_SCHEDULE");
  const bool glue_e9 = experiment_flag("TFS_GLUE_E9_INT32_COLIDX");
  const bool fwd_v2 = experiment_flag("TFS_FWD_V2_SINGLE_SCAN");
  const bf16* xp = reinterpret_cast<const bf16*>(
      x.data_ptr<at::BFloat16>());
  const std::int64_t* rp = rp_t.data_ptr<std::int64_t>();
  const std::int64_t* ci = ci_t.data_ptr<std::int64_t>();
  bool e9_index_reused = false;
  const bool use_e9 = formal_colidx_enabled(glue_e9, ci, ci_t.numel());
  const std::int32_t* ci32 = use_e9
      ? int32_colidx_workspace(ci, ci_t.numel(), e9_index_reused) : nullptr;

  auto pulled = at::empty({n, k}, x.options());
  bf16* dst = reinterpret_cast<bf16*>(
      pulled.data_ptr<at::BFloat16>());
  std::vector<std::vector<int>> own_local;
  bool schedule_reused = false;
  const bool locality_schedule = locality_schedule_enabled(n, threads);
  const std::vector<std::vector<int>>* own_ptr = nullptr;
  if (glue_e7) {
    own_ptr = &forward_schedule_workspace(n, threads, panel, rp, ci,
                                          schedule_reused);
  } else {
    own_local = locality_schedule
        ? panel_schedule_source_reuse(rp, ci, n, panel, threads)
        : panel_schedule(rp, n, panel, threads);
    own_ptr = &own_local;
  }
  const auto& own = *own_ptr;
  parallel_workers(threads, [&](int tid) {
    for (int p : own[tid]) {
      const int r0 = p * panel;
      const int v = std::min(panel, n - r0);
      pull_panel(rp, ci, ci32, use_e9, xp, k, k, r0, v,
                 dst + static_cast<std::size_t>(r0) * k, k,
                 glue_e2, fwd_v2);
    }
  });

  if (internal_profile_enabled())
    std::cout << std::fixed << std::setprecision(6)
              << "TFS_INTERNAL,kind=pull_only_bf16,k=" << k
              << ",threads=" << threads
              << ",glue_e2=" << (glue_e2 ? 1 : 0)
              << ",glue_e7=" << (glue_e7 ? 1 : 0)
              << ",glue_e9=" << (glue_e9 ? 1 : 0)
              << ",e9_index_reused=" << (e9_index_reused ? 1 : 0)
              << ",schedule_reused=" << (schedule_reused ? 1 : 0)
              << ",total_ms=" << (now_ms() - profile_t0) << std::endl;
  return pulled;
}

// Fused row scaling and FP32->BF16 conversion for the wide backward.  The
// framework expression (grad * scale.unsqueeze(1)).to(BF16) materializes a
// full FP32 temporary before writing the BF16 gradient.  This kernel performs
// the same multiply and round-to-nearest-even conversion directly into the
// BF16 destination.
at::Tensor c3_scale_grad_bf16_v1(
    const at::Tensor& grad_in, const at::Tensor& scale_in, int64_t threads64) {
  auto grad=grad_in.contiguous(),scale=scale_in.contiguous();
  TORCH_CHECK(grad.device().is_cpu() && grad.scalar_type()==at::kFloat &&
              scale.device().is_cpu() && scale.scalar_type()==at::kFloat,
              "fused grad scaling expects CPU FP32 tensors");
  TORCH_CHECK(grad.dim()==2 && scale.dim()==1 &&
              scale.numel()==grad.size(0),
              "fused grad scaling shape mismatch");
  const int n=static_cast<int>(grad.size(0));
  const int d=static_cast<int>(grad.size(1));
  const int threads=static_cast<int>(threads64);
  TORCH_CHECK(n>=1 && d>=1 && threads>=1 && threads<=32,
              "fused grad scaling shape unsupported");
  auto out=at::empty({n,d},grad.options().dtype(at::kBFloat16));
  const float* gp=grad.data_ptr<float>();
  const float* sp=scale.data_ptr<float>();
  bf16* op=reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  parallel_range(n,threads,[&](std::int64_t begin,std::int64_t end){
    for(std::int64_t i=begin;i<end;++i){
      const float* gr=gp+static_cast<std::size_t>(i)*d;
      bf16* orow=op+static_cast<std::size_t>(i)*d;
      const __m512 sv=_mm512_set1_ps(sp[i]);
      int q=0;
      for(;q+32<=d;q+=32){
        const __m512 v0=_mm512_mul_ps(_mm512_loadu_ps(gr+q),sv);
        const __m512 v1=_mm512_mul_ps(_mm512_loadu_ps(gr+q+16),sv);
        _mm512_storeu_si512(reinterpret_cast<void*>(orow+q),
            (__m512i)_mm512_cvtne2ps_pbh(v1,v0));
      }
      for(;q<d;++q)orow[q]=to_bf16(gr[q]*sp[i]);
    }
  });
  return out;
}

// Fused wide-backward gradient preparation.  In addition to the BF16 scaled
// gradient, accumulate db while each row is already resident in cache.  The
// per-worker FP32 slabs preserve a deterministic reduction order and avoid
// atomics on the hot row loop.  This is opt-in because some very small d
// shapes can still prefer ATen's highly tuned standalone reduction.
std::vector<at::Tensor> c3_scale_grad_bf16_db_v2(
    const at::Tensor& grad_in, const at::Tensor& scale_in, int64_t threads64) {
  auto grad=grad_in.contiguous(),scale=scale_in.contiguous();
  TORCH_CHECK(grad.device().is_cpu() && grad.scalar_type()==at::kFloat &&
              scale.device().is_cpu() && scale.scalar_type()==at::kFloat &&
              grad.dim()==2 && scale.dim()==1 &&
              scale.numel()==grad.size(0),
              "fused grad/db expects CPU FP32 tensors");
  const int n=static_cast<int>(grad.size(0));
  const int d=static_cast<int>(grad.size(1));
  const int threads=static_cast<int>(threads64);
  TORCH_CHECK(n>=1 && d>=1 && threads>=1 && threads<=32,
              "fused grad/db shape unsupported");
  auto out=at::empty({n,d},grad.options().dtype(at::kBFloat16));
  auto db=at::zeros({d},grad.options());
  std::vector<float> partial(static_cast<std::size_t>(threads)*d,0.0f);
  const float* gp=grad.data_ptr<float>();
  const float* sp=scale.data_ptr<float>();
  bf16* op=reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  parallel_workers(threads,[&](int tid){
    const int begin=n*tid/threads, end=n*(tid+1)/threads;
    float* local=partial.data()+static_cast<std::size_t>(tid)*d;
    for(int i=begin;i<end;++i){
      const float* gr=gp+static_cast<std::size_t>(i)*d;
      bf16* orow=op+static_cast<std::size_t>(i)*d;
      const float sv=sp[i];
      const __m512 svv=_mm512_set1_ps(sv);
      int q=0;
      for(;q+32<=d;q+=32){
        const __m512 v0=_mm512_mul_ps(_mm512_loadu_ps(gr+q),svv);
        const __m512 v1=_mm512_mul_ps(_mm512_loadu_ps(gr+q+16),svv);
        _mm512_storeu_si512(reinterpret_cast<void*>(orow+q),
            (__m512i)_mm512_cvtne2ps_pbh(v1,v0));
        _mm512_storeu_ps(local+q,
            _mm512_add_ps(_mm512_loadu_ps(local+q),
                          _mm512_loadu_ps(gr+q)));
        _mm512_storeu_ps(local+q+16,
            _mm512_add_ps(_mm512_loadu_ps(local+q+16),
                          _mm512_loadu_ps(gr+q+16)));
      }
      for(;q<d;++q){
        orow[q]=to_bf16(gr[q]*sv);
        local[q]+=gr[q];
      }
    }
  });
  float* dbp=db.data_ptr<float>();
  for(int tid=0;tid<threads;++tid){
    const float* local=partial.data()+static_cast<std::size_t>(tid)*d;
    for(int j=0;j<d;++j)dbp[j]+=local[j];
  }
  if(internal_profile_enabled())
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_SCALE_DB rows="<<n<<",d="<<d<<",threads="<<threads
      <<",partial_bytes="<<(partial.size()*sizeof(float))<<std::endl;
  return {out,db};
}

// Generic aggregate-saved backward.  The forward has already materialized
// P=B*Hs in BF16.  dW therefore needs no second sparse traversal; when dX is
// requested, only the mathematically necessary pull of dP is performed.  The
// API intentionally covers D<=128 first so it can use the same AMX pull-only
// primitive and remains independent of dataset names or class counts.
std::vector<at::Tensor> c3_backward_aggregate_saved_amx_v4(
    const at::Tensor& grad_in, const at::Tensor& pulled_in,
    const at::Tensor& weight_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, bool compute_dx) {
  const double t0=now_ms();
  auto grad=grad_in.contiguous(),pulled=pulled_in.contiguous();
  auto weight=weight_in.contiguous(),rp=rowptr_in.contiguous();
  auto ci=colidx_in.contiguous(),scale=scale_in.contiguous();
  TORCH_CHECK(grad.device().is_cpu() && grad.scalar_type()==at::kFloat &&
              pulled.device().is_cpu() && pulled.scalar_type()==at::kBFloat16 &&
              weight.device().is_cpu() && weight.scalar_type()==at::kFloat &&
              scale.device().is_cpu() && scale.scalar_type()==at::kFloat &&
              rp.device().is_cpu() && rp.scalar_type()==at::kLong &&
              ci.device().is_cpu() && ci.scalar_type()==at::kLong,
              "aggregate-saved-v4 expects contiguous CPU tensors");
  TORCH_CHECK(grad.dim()==2 && pulled.dim()==2 && weight.dim()==2 &&
              scale.dim()==1 && rp.dim()==1 && ci.dim()==1,
              "aggregate-saved-v4 rank contract failed");
  const int n=static_cast<int>(grad.size(0));
  const int d=static_cast<int>(grad.size(1));
  const int k=static_cast<int>(weight.size(0));
  const int threads=static_cast<int>(threads64);
  TORCH_CHECK(n>=1 && k>=1 && d>=1 && d<=128 && threads>=1 && threads<=32 &&
              pulled.size(0)==n && pulled.size(1)==k && weight.size(1)==d &&
              rp.numel()==n+1 && scale.numel()==n,
              "aggregate-saved-v4 shape unsupported");

  const double scale0=now_ms();
  at::Tensor gs,db;
  if(experiment_flag("TFS_SCALE_GRAD_DB_NATIVE")) {
    auto scaled_db=c3_scale_grad_bf16_db_v2(grad,scale,threads);
    gs=scaled_db[0]; db=scaled_db[1];
  } else {
    gs=c3_scale_grad_bf16_v1(grad,scale,threads);
    db=grad.sum(0);
  }
  const double scale1=now_ms();
  auto dw=at::matmul(pulled.transpose(0,1).contiguous(),gs).to(at::kFloat);
  at::Tensor dx=at::empty({0},grad.options());
  double pull_ms=0.0;
  if(compute_dx){
    const double p0=now_ms();
    auto wb=weight.to(at::kBFloat16);
    auto dp=at::matmul(gs,wb.transpose(0,1).contiguous()).to(at::kFloat);
    auto dh=c3_pull_only_amx_v1(dp,rp,ci,threads);
    dx=dh*scale.unsqueeze(1);
    pull_ms=now_ms()-p0;
  }
  auto meta=at::tensor({threads,512,round_up(d,32),round_up(k,64)},
      at::TensorOptions().dtype(at::kLong));
  if(internal_profile_enabled()){
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_AGG_SAVED compute_dx="<<(compute_dx?1:0)
      <<",pulled_bytes="<<(pulled.numel()*pulled.element_size())
      <<",backward_sparse_calls="<<(compute_dx?1:0)
      <<",scale_grad_ms="<<(scale1-scale0)
      <<",dw_ms="<<(now_ms()-scale1)
      <<",pull_ms="<<pull_ms
      <<",total_ms="<<(now_ms()-t0)<<std::endl;
  }
  return {dx,dw,db,meta};
}

// Wide-output AMX C3 path.  Keep the established transform-first numerical
// order, but move the output-column tiling inside the extension.  Hs, BF16
// weight conversion, CSR schedule construction, and Python dispatch are each
// performed once; the sparse pull and epilogue still use the same AMX tile
// contract (D<=128) for every internal column block.
static std::vector<at::Tensor> c3_forward_wide_amx_impl(
    const at::Tensor& x_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, const at::Tensor* cached_hs_in) {
  auto x=x_in.contiguous(),w=weight_in.contiguous(),b=bias_in.contiguous();
  auto rp_t=rowptr_in.contiguous(),ci_t=colidx_in.contiguous(),s=scale_in.contiguous();
  TORCH_CHECK(x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat &&
              b.scalar_type()==at::kFloat && s.scalar_type()==at::kFloat,
              "wide AMX forward expects FP32 tensors");
  TORCH_CHECK(rp_t.scalar_type()==at::kLong && ci_t.scalar_type()==at::kLong,
              "wide AMX CSR must be int64");
  const int n=static_cast<int>(x.size(0));
  const int k=static_cast<int>(x.size(1));
  const int d=static_cast<int>(w.size(1));
  const int threads=static_cast<int>(threads64),panel=512;
  TORCH_CHECK(w.size(0)==k && b.numel()==d && s.numel()==n &&
              k>=1 && d>128 && threads>=1 && threads<=32,
              "wide AMX forward shape unsupported");
  const bool glue_e2=experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bool glue_e4=experiment_flag("TFS_GLUE_E4_FUSED_EPILOGUE");
  const bool glue_e5=experiment_flag("TFS_GLUE_E5_VEC_HS");
  const bool glue_e7=experiment_flag("TFS_GLUE_E7_FORWARD_SCHEDULE");
  const bool glue_e9=experiment_flag("TFS_GLUE_E9_INT32_COLIDX");
  const bool fwd_v2=experiment_flag("TFS_FWD_V2_SINGLE_SCAN");
  const float* xp=x.data_ptr<float>();
  const float* sp=s.data_ptr<float>();
  const std::int64_t* rp=rp_t.data_ptr<std::int64_t>();
  const std::int64_t* ci=ci_t.data_ptr<std::int64_t>();
  bool e9_index_reused=false;
  const bool use_e9=formal_colidx_enabled(glue_e9,ci,ci_t.numel());
  const std::int32_t* ci32=use_e9?
      int32_colidx_workspace(ci,ci_t.numel(),e9_index_reused):nullptr;
  at::Tensor hs;
  if (cached_hs_in != nullptr) {
    TORCH_CHECK(cached_hs_in->device().is_cpu() &&
                cached_hs_in->scalar_type() == at::kBFloat16 &&
                cached_hs_in->dim() == 2 && cached_hs_in->is_contiguous() &&
                cached_hs_in->size(0) == n && cached_hs_in->size(1) == k,
                "cached wide Hs must be contiguous CPU BF16 with x shape");
    hs = *cached_hs_in;
  } else {
    hs = build_hs_bf16_exact(x, s, threads);
  }
  bf16* hsp=reinterpret_cast<bf16*>(hs.data_ptr<at::BFloat16>());
  auto wb=w.to(at::kBFloat16);
  auto out=at::empty({n,d},x.options());
  bool schedule_reused=false;
  const bool locality_schedule=locality_schedule_enabled(n,threads);
  const std::vector<std::vector<int>>* own_ptr=nullptr;
  std::vector<std::vector<int>> own_local;
  if(glue_e7)own_ptr=&forward_schedule_workspace(n,threads,panel,rp,ci,
                                                  schedule_reused);
  else {own_local=locality_schedule?
              panel_schedule_source_reuse(rp,ci,n,panel,threads):
              panel_schedule(rp,n,panel,threads);own_ptr=&own_local;}
  const auto& own=*own_ptr;
  float* outp=out.data_ptr<float>();
  const float* bp=b.data_ptr<float>();
  // Transform-first High-D uses one logical CSR traversal per macro D slab,
  // instead of repeating the traversal for every 128-column AMX tile.  The
  // dense temporary is bounded by the slab width and the shape-only planner
  // may override it through TFS_TRANSFORM_HIGHD_D_TILE.
  int transform_d_tile = 128;
  if (d > 128) {
    transform_d_tile = 256;
    if (const char* value = std::getenv("TFS_TRANSFORM_HIGHD_D_TILE")) {
      char* end = nullptr;
      const long requested = std::strtol(value, &end, 10);
      TORCH_CHECK(end != value && *end == '\0' && requested >= 128 &&
                      requested <= 2048 && (requested % 32) == 0,
                  "TFS_TRANSFORM_HIGHD_D_TILE must be a 32-aligned value in [128,2048]");
      transform_d_tile = static_cast<int>(requested);
    }
  }
  for(int start=0;start<d;start+=transform_d_tile){
    const int td=std::min(transform_d_tile,d-start);
    auto wt=wb.narrow(1,start,td).contiguous();
    auto transformed=at::matmul(hs,wt).contiguous();
    auto aggregated=at::empty({n,td},hs.options());
    const bf16* src=reinterpret_cast<const bf16*>(
        transformed.data_ptr<at::BFloat16>());
    bf16* dst=reinterpret_cast<bf16*>(aggregated.data_ptr<at::BFloat16>());
    parallel_workers(threads,[&](int tid){
      for(int p:own[tid]){
        const int r0=p*panel,v=std::min(panel,n-r0);
        pull_panel(rp,ci,ci32,use_e9,src,td,td,r0,v,
                   dst+static_cast<std::size_t>(r0)*td,td,
                   glue_e2,fwd_v2);
      }
    });
    const bf16* aggp=reinterpret_cast<const bf16*>(
        aggregated.data_ptr<at::BFloat16>());
    parallel_range(n,threads,[&](std::int64_t begin,std::int64_t end){
      for(std::int64_t i=begin;i<end;++i){
        const __m512 sv=_mm512_set1_ps(sp[i]);
        const bf16* ar=aggp+static_cast<std::size_t>(i)*td;
        float* orow=outp+static_cast<std::size_t>(i)*d+start;
        const float* br=bp+start;
        int q=0;
        if(glue_e4){
          for(;q+16<=td;q+=16){
            const __m512 v=load_bf16x16_fp32(ar+q);
            _mm512_storeu_ps(orow+q,
                _mm512_fmadd_ps(v,sv,_mm512_loadu_ps(br+q)));
          }
        }
        for(;q<td;++q)orow[q]=from_bf16(ar[q])*sp[i]+br[q];
      }
    });
  }
  if (internal_profile_enabled()) {
    std::cout << "TFS_TRANSFORM_HIGHD_FORWARD N=" << n << ",K=" << k
              << ",D=" << d << ",d_tile=" << transform_d_tile
              << ",slabs=" << ((d + transform_d_tile - 1) /
                                   transform_d_tile)
              << std::endl;
  }
  return {out,hs};
}

std::vector<at::Tensor> c3_forward_wide_amx_v3(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads) {
  return c3_forward_wide_amx_impl(x, weight, bias, rowptr, colidx, scale,
                                  threads, nullptr);
}

std::vector<at::Tensor> c3_forward_wide_cached_hs_amx_v1(
    const at::Tensor& x, const at::Tensor& hs,
    const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads) {
  TORCH_CHECK(x.is_contiguous() && hs.is_contiguous() &&
              weight.is_contiguous() && bias.is_contiguous() &&
              rowptr.is_contiguous() && colidx.is_contiguous() &&
              scale.is_contiguous(),
              "cached wide Hs consumer requires contiguous inputs");
  return c3_forward_wide_amx_impl(x, weight, bias, rowptr, colidx, scale,
                                  threads, &hs);
}

std::vector<at::Tensor> c3_forward_saved_t_amx_v3(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& rowptr, const at::Tensor& colidx,
    const at::Tensor& scale, int64_t threads) {
  return c3_forward_amx_impl(x,weight,bias,rowptr,colidx,scale,threads,
                             false,true,false);
}

std::vector<at::Tensor> c3_backward_amx_v2(
    const at::Tensor& grad_in, const at::Tensor& hs_in,
    const at::Tensor& weight_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, bool compute_dx) {
  const double profile_t0=now_ms();
  TORCH_CHECK(grad_in.device().is_cpu() && grad_in.scalar_type()==at::kFloat,
              "AMX-v2 grad must be CPU FP32");
  TORCH_CHECK((hs_in.scalar_type()==at::kFloat || hs_in.scalar_type()==at::kBFloat16) && weight_in.scalar_type()==at::kFloat && scale_in.scalar_type()==at::kFloat,
              "AMX-v2 Hs must be FP32/BF16 and weight/scale FP32");
  TORCH_CHECK(rowptr_in.scalar_type()==at::kLong && colidx_in.scalar_type()==at::kLong,
              "AMX-v2 CSR must be int64");
  auto grad=grad_in.contiguous(),hs=hs_in.contiguous(),w=weight_in.contiguous();
  auto rp_t=rowptr_in.contiguous(),ci_t=colidx_in.contiguous(),s=scale_in.contiguous();
  TORCH_CHECK(w.dim() == 2 && hs.dim() == 2,
              "AMX-v2 backward expects rank-2 Hs and weight");
  // `k` is the logical feature width carried by the weight.  A V2 padded Hs
  // may have a wider physical row stride, but its tail is not part of dW/dH.
  const int n=static_cast<int>(grad.size(0)),d=static_cast<int>(grad.size(1)),k=static_cast<int>(w.size(0));
  const int hs_width=static_cast<int>(hs.size(1));
  const int threads=static_cast<int>(threads64),dp=round_up(d,32),kp=round_up(k,64),panel=512;
  const bool glue_e1=experiment_flag("TFS_GLUE_E1_FUSED_DB");
  const bool glue_e11=experiment_flag("TFS_GLUE_E11_VEC_GRAD_DB");
  const bool glue_e12=experiment_flag("TFS_GLUE_E12_D47_SINGLE_SCAN");
  const bool glue_e13=experiment_flag("TFS_GLUE_E13_ACTIVE_ROW");
  const bool glue_e14=experiment_flag("TFS_GLUE_E14_ACTIVE_ROW_WIDE");
  const bool glue_e2=experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bool glue_e3=experiment_flag("TFS_GLUE_E3_EMPTY_DX");
  const bool glue_e6_zero=experiment_flag("TFS_GLUE_E6_LOCAL_ZERO");
  const bool glue_e6_reduce=experiment_flag("TFS_GLUE_E6_PARALLEL_REDUCE");
  const bool glue_e6_wt=experiment_flag("TFS_GLUE_E6_SERIAL_WT");
  const bool glue_e9=experiment_flag("TFS_GLUE_E9_INT32_COLIDX");
  const bool formal_single_scan=formal_mode_enabled(
      "TFS_SMALL_SINGLE_SCAN", d<=128);
  const bool formal_active_row=formal_mode_enabled(
      "TFS_ACTIVE_ROW", d<=128);
  TORCH_CHECK(hs.size(0)==n && (hs_width==k || hs_width==kp) &&
              w.size(1)==d && rp_t.numel()==n+1 && s.numel()==n,
              "AMX-v2 shape mismatch");
  // dh/dw kernels already iterate over k_padded tiles; this probe removes
  // only the legacy K cap and retains the D/worker safety bounds.
  TORCH_CHECK(k>=1 && d<=128 && threads>=1 && threads<=32,"AMX-v2 wide-K backward supports D<=128 and 1..32 threads");
  const float* gp=grad.data_ptr<float>();const float* hp=hs.scalar_type()==at::kFloat?hs.data_ptr<float>():nullptr;const float* wp=w.data_ptr<float>();const float* sp=s.data_ptr<float>();
  const bf16* hs_bf=hs.scalar_type()==at::kBFloat16?reinterpret_cast<const bf16*>(hs.data_ptr<at::BFloat16>()):nullptr;
  const std::int64_t* rp=rp_t.data_ptr<std::int64_t>();const std::int64_t* ci=ci_t.data_ptr<std::int64_t>();
  bool e9_index_reused=false;
  const bool use_e9=formal_colidx_enabled(glue_e9,ci,ci_t.numel());
  const std::int32_t* ci32=use_e9?
      int32_colidx_workspace(ci,ci_t.numel(),e9_index_reused):nullptr;
  const int hs_stride=static_cast<int>(hs.stride(0));
  // Compute the dataflow contract before looking up a workspace.  These
  // booleans are part of the cache key, so an entry can never be reused with
  // a buffer set that is too small (or needlessly maximal).
  const bool direct_hs=hs_bf!=nullptr && hs_stride>=hs_width;
  const bool vectorized_d128 = glue_e1 && glue_e11 && d==128;
  const bool build_active=((formal_active_row && d<=128) ||
                           (glue_e13 && d==47) || (glue_e14 && d==128)) &&
                          !vectorized_d128;
  const double profile_setup=now_ms();
  std::unique_lock<std::mutex> workspace_lock(backward_workspace_mutex());
  bool workspace_reused=false;
  BackwardWorkspaceSpec workspace_spec{
      n,d,k,dp,kp,threads,panel,rp,ci,false,direct_hs,compute_dx,
      build_active,glue_e1,true,BackwardImplKind::standard_c3};
  bool workspace_transient=false;
  BackwardWorkspace& ws=backward_workspace(workspace_spec,workspace_reused,workspace_transient);
  const double profile_stage_alloc=now_ms();
  // BF16 Hs can be consumed directly even when K is a non-AMX tail (for
  // example K=100).  The panel packer writes only logical columns and
  // zero-fills the padded tile, so avoid materialising a full N x Kp hb copy.
  // FP32 Hs still uses the converted workspace because its conversion order
  // must be fixed before worker kernels read it.
  // The fused E1/E11 D=128 preparation below is fully vectorized and does
  // not materialize the per-row support byte.  Do not inspect the reusable
  // workspace's uninitialized ``active`` buffer on that path: besides being
  // a correctness hazard, the old unconditional count added a full-N scan
  // even though the density gate could never be used.  The scalar/general
  // preparation paths still build the mask and retain the sparse-row gate.
  // E1 already visits every row while producing the scaled gradient.  Keep
  // the active-row population count in the same worker pass instead of
  // launching a second full-N ``std::count`` scan on the dense cases.
  std::vector<std::int64_t> active_local(
      (build_active && glue_e1) ? static_cast<std::size_t>(threads) : 0, 0);
  if (glue_e1 && glue_e11 && d==128) {
    parallel_workers(threads,[&](int tid){
      __m512 dbv[8];
      for(int j=0;j<8;++j)dbv[j]=_mm512_setzero_ps();
      for(int p:ws.own[tid]) for(std::int64_t i=static_cast<std::int64_t>(p)*panel,
          end=std::min<std::int64_t>(n,i+panel);i<end;++i){
        const __m512 sv=_mm512_set1_ps(sp[i]);
        const float* gr=gp+static_cast<std::size_t>(i)*128;
        bf16* gsrow=ws.gs.data()+static_cast<std::size_t>(i)*dp;
        for(int q=0;q<128;q+=32){
          const __m512 v0=_mm512_loadu_ps(gr+q);
          const __m512 v1=_mm512_loadu_ps(gr+q+16);
          dbv[q/16]=_mm512_add_ps(dbv[q/16],v0);
          dbv[q/16+1]=_mm512_add_ps(dbv[q/16+1],v1);
          const __m512 s0=_mm512_mul_ps(v0,sv);
          const __m512 s1=_mm512_mul_ps(v1,sv);
          _mm512_storeu_si512(reinterpret_cast<void*>(gsrow+q),
              (__m512i)_mm512_cvtne2ps_pbh(s1,s0));
        }
        if(!direct_hs) for(int q=0;q<k;++q)
          ws.hb.data()[static_cast<std::size_t>(i)*kp+q]=
              hs_bf?hs_bf[static_cast<std::size_t>(i)*hs_stride+q]:
                    to_bf16(hp[static_cast<std::size_t>(i)*hs_stride+q]);
      }
      float* dbp=ws.db_local.data()+static_cast<std::size_t>(tid)*dp;
      for(int j=0;j<8;++j)_mm512_storeu_ps(dbp+j*16,dbv[j]);
    });
  } else if (glue_e1) {
    parallel_workers(threads,[&](int tid){
      float* dbp=ws.db_local.data()+static_cast<std::size_t>(tid)*dp;
      std::fill(dbp,dbp+dp,0.0f);
      std::int64_t active_rows_local=0;
      for(int p:ws.own[tid]) for(std::int64_t i=static_cast<std::int64_t>(p)*panel,
          end=std::min<std::int64_t>(n,i+panel);i<end;++i){
        bf16 support=0;
        for(int q=0;q<d;++q){
          const float g=gp[static_cast<std::size_t>(i)*d+q];
          dbp[q]+=g;
          const bf16 scaled=to_bf16(g*sp[i]);
          ws.gs.data()[static_cast<std::size_t>(i)*dp+q]=scaled;
          support=static_cast<bf16>(support|scaled);
        }
        if(build_active){
          const bool row_active=support!=0;
          ws.active.data()[i]=row_active;
          active_rows_local+=row_active?1:0;
        }
        if(!direct_hs) for(int q=0;q<k;++q)ws.hb.data()[static_cast<std::size_t>(i)*kp+q]=hs_bf?hs_bf[static_cast<std::size_t>(i)*hs_stride+q]:to_bf16(hp[static_cast<std::size_t>(i)*hs_stride+q]);
      }
      if(build_active)active_local[static_cast<std::size_t>(tid)]=active_rows_local;
    });
  } else {
    parallel_workers(threads,[&](int tid){
      for(int p:ws.own[tid]) for(std::int64_t i=static_cast<std::int64_t>(p)*panel,
          end=std::min<std::int64_t>(n,i+panel);i<end;++i){
        bf16 support=0;
        for(int q=0;q<d;++q){
          const bf16 scaled=to_bf16(gp[static_cast<std::size_t>(i)*d+q]*sp[i]);
          ws.gs.data()[static_cast<std::size_t>(i)*dp+q]=scaled;
          support=static_cast<bf16>(support|scaled);
        }
        if(build_active)ws.active.data()[i]=support!=0;
        if(!direct_hs) for(int q=0;q<k;++q)ws.hb.data()[static_cast<std::size_t>(i)*kp+q]=hs_bf?hs_bf[static_cast<std::size_t>(i)*hs_stride+q]:to_bf16(hp[static_cast<std::size_t>(i)*hs_stride+q]);
      }
    });
  }
  const double profile_stage_copy=now_ms();
  std::int64_t active_count=n;
  double active_density=1.0;
  if(build_active){
    if(glue_e1){
      active_count=std::accumulate(active_local.begin(),active_local.end(),
                                   std::int64_t{0});
    } else {
      active_count=std::count(ws.active.data(),ws.active.data()+n,
                              static_cast<std::uint8_t>(1));
    }
    active_density=static_cast<double>(active_count)/n;
  }
  double active_threshold=0.25;
  if(const char* value=std::getenv("TFS_E13_ACTIVE_MAX_DENSITY"))
    active_threshold=std::strtod(value,nullptr);
  TORCH_CHECK(active_threshold>=0.0 && active_threshold<=1.0,
              "TFS_E13_ACTIVE_MAX_DENSITY must be in [0,1]");
  const bool use_active=build_active && active_density<=active_threshold;
  const bool use_single_scan=(formal_single_scan || glue_e12 || use_active) &&
                             d<=128;
  if(compute_dx){
    if(glue_e6_wt && d<=256){
      for(int q=0;q<d;++q)for(int p=0;p<k;++p)
        ws.wt.data()[static_cast<std::size_t>(q)*kp+p]=to_bf16(wp[static_cast<std::size_t>(p)*d+q]);
    } else {
      parallel_range(d,threads,[&](std::int64_t begin,std::int64_t end){
        for(std::int64_t q=begin;q<end;++q)for(int p=0;p<k;++p)
          ws.wt.data()[static_cast<std::size_t>(q)*kp+p]=to_bf16(wp[static_cast<std::size_t>(p)*d+q]);
      });
    }
  }
  const double profile_wt=now_ms();
  if(compute_dx)pack_rhs_into(ws.wt.data(),dp,kp,ws.packed_wt.data());
  const double profile_pack=now_ms();
  // AMX stores complete 16-column tiles.  For a logical K tail (for example
  // ogbn-products K=100), writing those tiles directly into an N x K tensor
  // overruns each row.  Keep the kernel's padded stride private and compact
  // only the logical columns after all worker writes complete.
  auto dx_padded=compute_dx?(glue_e3?at::empty({n,kp},grad.options()):at::zeros({n,kp},grad.options())):at::empty({0},grad.options());
  auto dw=at::zeros({k,d},grad.options());
  at::Tensor db;
  if (glue_e1) db=at::empty({d},grad.options());
  const double profile_output_alloc=now_ms();
  if (glue_e1) {
    float* db_out=db.data_ptr<float>();
    for(int q=0;q<d;++q){
      float sum=0.0f;
      for(int tid=0;tid<threads;++tid)
        sum+=ws.db_local.data()[static_cast<std::size_t>(tid)*dp+q];
      db_out[q]=sum;
    }
  } else db=grad.sum(0);
  const double profile_db=now_ms();
  if(!glue_e6_zero)std::fill(ws.local.data(),ws.local.data()+ws.local.size(),0.0f);
  const double profile_scratch=now_ms();
  const auto& own=ws.own;float* dxp=compute_dx?dx_padded.data_ptr<float>():nullptr;
  const bf16* hb_base=direct_hs?hs_bf:ws.hb.data();
  const double profile_schedule=now_ms();
  std::vector<double> y_ms(threads),dh_ms(threads),t2_ms(threads),dw_ms(threads),thread_wall_ms(threads);
  std::vector<double> entry_delay_ms(threads),config_ms(threads),release_ms(threads);
  std::vector<int> cpu_id(threads,-1);
  const double profile_kernel0=now_ms();
  parallel_workers(threads,[&](int tid){
    if(glue_e6_zero){
      float* slab=ws.local.data()+static_cast<std::size_t>(tid)*dp*kp;
      std::fill(slab,slab+static_cast<std::size_t>(dp)*kp,0.0f);
    }
    const double entry=now_ms();
    const int cpu=sched_getcpu();
    const double config0=now_ms();
    bv2::configure_amx_tiles_16x64();
    const double config1=now_ms();
    entry_delay_ms[tid]=entry-profile_kernel0;
    config_ms[tid]=config1-config0;
    cpu_id[tid]=cpu;
    {
      const double thread_t0=now_ms();Scratch& z=*ws.scratch[tid];
      for(int p:own[tid]){const int row0=p*panel,valid=std::min(panel,n-row0),rows=round_up(valid,32);
        if(rows>valid)std::fill(z.y.data()+static_cast<std::size_t>(valid)*dp,z.y.data()+static_cast<std::size_t>(rows)*dp,bf16(0));
        double q=now_ms();
        pull_panel(rp,ci,ci32,use_e9,ws.gs.data(),d,dp,row0,valid,
                   z.y.data(),dp,glue_e2,use_single_scan,
                   use_active?ws.active.data():nullptr);
        y_ms[tid]+=now_ms()-q;q=now_ms();
        if(compute_dx)bv2::dh_amx_4c2a2b(z.y.data(),rows,valid,dp,ws.packed_wt.data(),k,kp,dxp,row0,sp,true,nullptr);
        dh_ms[tid]+=now_ms()-q;q=now_ms();
        bv2::transpose_y_avx512_t2(z.y.data(),rows,dp,z.yt.data());
        t2_ms[tid]+=now_ms()-q;q=now_ms();
        if(hs_bf!=nullptr && hs_width!=kp)
          bv2::pack_h_panel_direct_tail(hs_bf,hs_stride,k,row0,valid,rows,kp,z.hp.data());
        else
          bv2::pack_h_panel_direct(hb_base,kp,row0,valid,rows,kp,z.hp.data());
        bv2::dw_amx_4c2a2b(z.yt.data(),dp,rows,z.hp.data(),k,kp,ws.local.data()+static_cast<std::size_t>(tid)*dp*kp,true,nullptr);
        dw_ms[tid]+=now_ms()-q;
      }
      thread_wall_ms[tid]=now_ms()-thread_t0;
    }
    const double release0=now_ms();
    _tile_release();
    const double release1=now_ms();
    release_ms[tid]=release1-release0;
  });
  const double profile_kernel1=now_ms();
  const int per_numa=runtime_per_numa(threads);
  if(glue_e6_reduce)
    reduce_dwt_parallel_deterministic(ws.local.data(),threads,per_numa,dp*kp,ws.dwt.data());
  else
    reduce_dwt_persistent(ws.local.data(),threads,per_numa,dp*kp,
                          ws.group_reduce.data(),ws.numa_reduce.data(),ws.dwt.data());
  const double profile_reduce=now_ms();
  float* dwp=dw.data_ptr<float>();for(int a=0;a<k;++a)for(int b=0;b<d;++b)dwp[static_cast<std::size_t>(a)*d+b]=ws.dwt.data()[static_cast<std::size_t>(b)*kp+a];
  auto dx=compute_dx?(k==kp?dx_padded:dx_padded.narrow(1,0,k).contiguous()):dx_padded;
  const double profile_end=now_ms();
  auto meta=at::tensor({threads,512,dp,kp},at::TensorOptions().dtype(at::kLong));
  if(internal_profile_enabled()) {
    auto maxv=[](const std::vector<double>& v){return *std::max_element(v.begin(),v.end());};
    auto minv=[](const std::vector<double>& v){return *std::min_element(v.begin(),v.end());};
    std::set<int> unique_cpus;for(int cpu:cpu_id)if(cpu>=0)unique_cpus.insert(cpu);
    for(int tid=0;tid<threads;++tid){
      const int cpu=cpu_id[tid];
      std::cout<<"TFS_NUMA worker="<<tid<<",cpu="<<cpu
        <<",socket="<<cpu_sysfs_int(cpu,"physical_package_id")
        <<",numa="<<cpu_numa_node(cpu)
        <<",core="<<cpu_sysfs_int(cpu,"core_id")<<std::endl;
    }
    std::cout<<"TFS_SPARSE D="<<d<<",single_scan="<<(use_single_scan?1:0)
      <<",blocks="<<((d+15)/16)<<",active_density="<<active_density
      <<",colidx=int32:"<<(use_e9?1:0)<<",panel="<<panel<<std::endl;
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_INTERNAL,kind=backward,k="<<k<<",d="<<d<<",threads="<<threads
      <<",compute_dx="<<(compute_dx?1:0)
      <<",glue_e1="<<(glue_e1?1:0)
      <<",glue_e11="<<(glue_e11?1:0)
      <<",glue_e12="<<(glue_e12?1:0)
      <<",glue_e13="<<(glue_e13?1:0)
      <<",glue_e14="<<(glue_e14?1:0)
      <<",formal_single_scan="<<(formal_single_scan?1:0)
      <<",formal_active_row="<<(formal_active_row?1:0)
      <<",locality_schedule="<<(ws.locality?1:0)
      <<",use_single_scan="<<(use_single_scan?1:0)
      <<",use_active="<<(use_active?1:0)
      <<",active_rows="<<active_count
      <<",active_density="<<active_density
      <<",glue_e2="<<(glue_e2?1:0)
      <<",glue_e3="<<(glue_e3?1:0)
      <<",glue_e6_zero="<<(glue_e6_zero?1:0)
      <<",glue_e6_reduce="<<(glue_e6_reduce?1:0)
      <<",glue_e6_wt="<<(glue_e6_wt?1:0)
      <<",glue_e9="<<(glue_e9?1:0)
      <<",use_e9="<<(use_e9?1:0)
      <<",e9_index_reused="<<(e9_index_reused?1:0)
      <<",per_numa="<<per_numa
      <<",setup_ms="<<(profile_setup-profile_t0)
      <<",workspace_reused="<<(workspace_reused?1:0)
      <<",workspace_transient="<<(workspace_transient?1:0)
      <<",workspace_cache_limit_bytes="<<workspace_cache_limit_bytes()
      <<",direct_hs="<<(direct_hs?1:0)
      <<",stage_alloc_zero_ms="<<(profile_stage_alloc-profile_setup)
      <<",grad_h_copy_ms="<<(profile_stage_copy-profile_stage_alloc)
      <<",wt_bf16_ms="<<(profile_wt-profile_stage_copy)
      <<",pack_wt_ms="<<(profile_pack-profile_wt)
      <<",output_alloc_ms="<<(profile_output_alloc-profile_pack)
      <<",db_ms="<<(profile_db-profile_output_alloc)
      <<",scratch_alloc_zero_ms="<<(profile_scratch-profile_db)
      <<",schedule_ms="<<(profile_schedule-profile_scratch)
      <<",kernel_wall_ms="<<(profile_kernel1-profile_kernel0)
      <<",callback_count="<<threads
      <<",unique_cpu_count="<<unique_cpus.size()
      <<",entry_delay_min_ms="<<minv(entry_delay_ms)
      <<",entry_delay_max_ms="<<maxv(entry_delay_ms)
      <<",config_min_ms="<<minv(config_ms)<<",config_max_ms="<<maxv(config_ms)
      <<",release_max_ms="<<maxv(release_ms)
      <<",y_thread_max_ms="<<maxv(y_ms)<<",dh_thread_max_ms="<<maxv(dh_ms)
      <<",t2_thread_max_ms="<<maxv(t2_ms)<<",dw_thread_max_ms="<<maxv(dw_ms)
      <<",thread_wall_min_ms="<<minv(thread_wall_ms)
      <<",thread_wall_max_ms="<<maxv(thread_wall_ms)
      <<",reduce_ms="<<(profile_reduce-profile_kernel1)
      <<",dw_scatter_ms="<<(profile_end-profile_reduce)
      <<",total_ms="<<(profile_end-profile_t0)<<std::endl;
  }
  return {dx,dw,db,meta};
}

// Wide-output backward wrapper.  The per-tile AMX backward kernel is kept
// unchanged for numerical compatibility; this wrapper moves the 24-way loop
// and gradient concatenation into native code and lets the caller reuse the
// single forward-produced Hs tensor.
std::vector<at::Tensor> c3_backward_wide_amx_v3(
    const at::Tensor& grad_in, const at::Tensor& hs_in,
    const at::Tensor& weight_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, bool compute_dx) {
  TORCH_CHECK(weight_in.dim()==2 && weight_in.size(1)>128,
              "wide AMX backward expects output D>128");
  const int d=static_cast<int>(weight_in.size(1));
  std::vector<at::Tensor> dw_parts;
  std::vector<at::Tensor> db_parts;
  dw_parts.reserve((d+127)/128);
  db_parts.reserve((d+127)/128);
  at::Tensor dx;
  for(int start=0;start<d;start+=128){
    const int td=std::min(128,d-start);
    auto result=c3_backward_amx_v2(
        grad_in.narrow(1,start,td).contiguous(), hs_in,
        weight_in.narrow(1,start,td).contiguous(), rowptr_in, colidx_in,
        scale_in, threads64, compute_dx);
    if(compute_dx){
      if(!dx.defined()) dx=result[0];
      else dx.add_(result[0]);
    }
    dw_parts.push_back(result[1]);
    db_parts.push_back(result[2]);
  }
  auto dw=at::cat(dw_parts,1);
  auto db=at::cat(db_parts,0);
  if(!compute_dx) dx=at::empty({0},grad_in.options());
  auto meta=at::tensor(std::vector<int64_t>{threads64,128,d},
      at::TensorOptions().dtype(at::kLong));
  return {dx,dw,db,meta};
}

// Aggregate-first high-D native stream.  Unlike c3_backward_wide_amx_v3
// above, this routine consumes the forward-produced pulled P=A(H) directly:
// dW=P^T(G*scale), dP=(G*scale)W^T, and only one sparse pull is performed for
// dX.  The existing wide wrapper cannot be used here because it consumes the
// pre-aggregation Hs tensor and would compute the wrong dW when passed P.
struct HighDAggregateStreamWorkspace {
  int dp, kp, threads, panel;
  bool compute_dx;
  std::size_t bytes_total = 0;
  double first_touch_ms = 0.0;
  bool first_touch_done = false;
  bool last_cache_hit = false;
  std::uint64_t last_used = 0;
  AlignedBuffer<float> local, dwt, group_reduce, numa_reduce;
  AlignedBuffer<float> db_local;
  AlignedBuffer<bf16> wt;
  AlignedBuffer<bf16> packed_wt;
  std::vector<std::unique_ptr<Scratch>> scratch;

  HighDAggregateStreamWorkspace(int dp_, int kp_, int threads_, int panel_,
                                bool compute_dx_)
      : dp(dp_), kp(kp_), threads(threads_), panel(panel_),
        compute_dx(compute_dx_),
        local(static_cast<std::size_t>(threads_) * dp_ * kp_),
        dwt(static_cast<std::size_t>(dp_) * kp_),
        group_reduce(static_cast<std::size_t>(
            (threads_ + runtime_reduce_group(threads_) - 1) /
            runtime_reduce_group(threads_)) * dp_ * kp_),
        numa_reduce(static_cast<std::size_t>(
            (threads_ + runtime_per_numa(threads_) - 1) /
            runtime_per_numa(threads_)) * dp_ * kp_),
        db_local(static_cast<std::size_t>(threads_) * dp_),
        wt(compute_dx ? static_cast<std::size_t>(dp_) * kp_ : 0),
        packed_wt(compute_dx ?
                  static_cast<std::size_t>(dp_ / 32) * (kp_ / 16) * 512 : 0) {
    scratch.reserve(static_cast<std::size_t>(threads_));
    for (int tid = 0; tid < threads_; ++tid)
      scratch.emplace_back(std::make_unique<Scratch>(panel_, dp_, kp_));
    bytes_total = local.size() * sizeof(float) + dwt.size() * sizeof(float) +
        group_reduce.size() * sizeof(float) +
        numa_reduce.size() * sizeof(float) + db_local.size() * sizeof(float) +
        wt.size() * sizeof(bf16) + packed_wt.size() * sizeof(bf16);
    for (const auto& ptr : scratch) {
      bytes_total += ptr->y.size() * sizeof(bf16) +
          ptr->yt.size() * sizeof(bf16) + ptr->hp.size() * sizeof(bf16);
    }
  }

  bool matches(int dp_, int kp_, int threads_, int panel_,
               bool compute_dx_) const {
    return dp == dp_ && kp == kp_ && threads == threads_ && panel == panel_ &&
           compute_dx == compute_dx_;
  }

  std::size_t scratch_bytes() const {
    std::size_t total = 0;
    for (const auto& ptr : scratch)
      total += ptr->y.size() * sizeof(bf16) +
          ptr->yt.size() * sizeof(bf16) + ptr->hp.size() * sizeof(bf16);
    return total;
  }

  // Diagnostic-only ownership map.  It records who first-touches/consumes
  // each shared allocation without changing the allocation or reduction
  // schedule.  The map is intentionally descriptive: a future NUMA dataflow
  // change must be benchmarked separately before any owner is re-laid out.
  void emit_buffer_profile() const {
    std::cout << "TFS_NUMA_BUFFER_PROFILE "
              << "{\"local\":{\"bytes\":"
              << local.size() * sizeof(float)
              << ",\"owner\":\"per_worker\",\"consumer\":\"worker\"},"
              << "\"dwt\":{\"bytes\":"
              << dwt.size() * sizeof(float)
              << ",\"owner\":\"striped\",\"consumer\":\"reduction\"},"
              << "\"group_reduce\":{\"bytes\":"
              << group_reduce.size() * sizeof(float)
              << ",\"owner\":\"striped\",\"consumer\":\"group_leader\"},"
              << "\"numa_reduce\":{\"bytes\":"
              << numa_reduce.size() * sizeof(float)
              << ",\"owner\":\"striped\",\"consumer\":\"numa_leader\"},"
              << "\"db_local\":{\"bytes\":"
              << db_local.size() * sizeof(float)
              << ",\"owner\":\"per_worker\",\"consumer\":\"worker\"},"
              << "\"wt\":{\"bytes\":"
              << wt.size() * sizeof(bf16)
              << ",\"owner\":\"striped\",\"consumer\":\"pack_readers\"},"
              << "\"packed_wt\":{\"bytes\":"
              << packed_wt.size() * sizeof(bf16)
              << ",\"owner\":\"striped\",\"consumer\":\"amx_readers\"},"
              << "\"scratch\":{\"bytes\":" << scratch_bytes()
              << ",\"owner\":\"per_worker\",\"consumer\":\"worker\"}}"
              << std::endl;
  }

  // Touch each allocation from the worker that owns it.  The old constructor
  // zero-filled every page on the thread that happened to create the cache
  // entry, defeating --localalloc on a multi-socket node.
  void first_touch() {
    if (first_touch_done) return;
    const double t0 = steady_ms_unconditional();
    auto worker_touch = [&](int tid) {
      auto touch_float = [&](float* ptr, std::size_t count) {
        if (count == 0) return;
        const std::size_t begin = count * static_cast<std::size_t>(tid) /
            static_cast<std::size_t>(threads);
        const std::size_t end = count * static_cast<std::size_t>(tid + 1) /
            static_cast<std::size_t>(threads);
        std::fill(ptr + begin, ptr + end, 0.0f);
      };
      auto touch_bf16 = [&](bf16* ptr, std::size_t count) {
        if (count == 0) return;
        const std::size_t begin = count * static_cast<std::size_t>(tid) /
            static_cast<std::size_t>(threads);
        const std::size_t end = count * static_cast<std::size_t>(tid + 1) /
            static_cast<std::size_t>(threads);
        std::fill(ptr + begin, ptr + end, bf16(0));
      };
      touch_float(local.data() + 0, local.size());
      touch_float(dwt.data() + 0, dwt.size());
      touch_float(group_reduce.data() + 0, group_reduce.size());
      touch_float(numa_reduce.data() + 0, numa_reduce.size());
      touch_float(db_local.data() + 0, db_local.size());
      touch_bf16(wt.data() + 0, wt.size());
      touch_bf16(packed_wt.data() + 0, packed_wt.size());
      // Scratch is private to one logical worker, so do not interleave its
      // pages across sockets.  The subsequent panel loop uses the same owner.
      Scratch& z = *scratch[static_cast<std::size_t>(tid)];
      std::fill(z.y.data(), z.y.data() + z.y.size(), bf16(0));
      std::fill(z.yt.data(), z.yt.data() + z.yt.size(), bf16(0));
      std::fill(z.hp.data(), z.hp.data() + z.hp.size(), bf16(0));
    };
    if (numa_first_touch_enabled(threads)) {
      parallel_workers(threads, worker_touch);
    } else {
      auto fill_float = [](AlignedBuffer<float>& buffer) {
        if (buffer.size())
          std::fill(buffer.data(), buffer.data() + buffer.size(), 0.0f);
      };
      auto fill_bf16 = [](AlignedBuffer<bf16>& buffer) {
        if (buffer.size())
          std::fill(buffer.data(), buffer.data() + buffer.size(), bf16(0));
      };
      fill_float(local); fill_float(dwt); fill_float(group_reduce);
      fill_float(numa_reduce); fill_float(db_local);
      fill_bf16(wt); fill_bf16(packed_wt);
      for (auto& ptr : scratch) {
        std::fill(ptr->y.data(), ptr->y.data() + ptr->y.size(), bf16(0));
        std::fill(ptr->yt.data(), ptr->yt.data() + ptr->yt.size(), bf16(0));
        std::fill(ptr->hp.data(), ptr->hp.data() + ptr->hp.size(), bf16(0));
      }
    }
    first_touch_ms = steady_ms_unconditional() - t0;
    first_touch_done = true;
  }
};

std::mutex& highd_stream_workspace_mutex() {
  static std::mutex mu;
  return mu;
}

// Scratch and deterministic reduction buffers are shared by repeated layer
// calls.  Match the existing backward workspace contract and serialize users
// so two concurrent autograd branches cannot overwrite a panel buffer.
std::mutex& highd_stream_runtime_mutex() {
  static std::mutex mu;
  return mu;
}

struct HighDWorkspaceCacheCounters {
  std::size_t hits = 0;
  std::size_t misses = 0;
  std::size_t evictions = 0;
  std::size_t bytes = 0;
  std::size_t limit = static_cast<std::size_t>(512) << 20;
};

HighDWorkspaceCacheCounters& highd_workspace_cache_counters() {
  static HighDWorkspaceCacheCounters counters;
  return counters;
}

HighDAggregateStreamWorkspace& highd_stream_workspace(
    int dp, int kp, int threads, int panel, bool compute_dx) {
  static std::vector<std::unique_ptr<HighDAggregateStreamWorkspace>> cache;
  static std::unique_ptr<HighDAggregateStreamWorkspace> transient;
  static std::size_t cache_bytes = 0;
  static std::uint64_t use_clock = 0;
  auto& counters = highd_workspace_cache_counters();
  std::lock_guard<std::mutex> lock(highd_stream_workspace_mutex());
  transient.reset();
  for (auto& ws : cache) {
    if (ws->matches(dp, kp, threads, panel, compute_dx)) {
      ws->last_used = ++use_clock;
      ws->last_cache_hit = true;
      ++counters.hits;
      return *ws;
    }
  }
  ++counters.misses;
  auto candidate = std::make_unique<HighDAggregateStreamWorkspace>(
      dp, kp, threads, panel, compute_dx);
  const std::size_t limit = workspace_cache_limit_bytes();
  candidate->last_used = ++use_clock;
  candidate->last_cache_hit = false;
  candidate->first_touch();
  if (candidate->bytes_total > limit) {
    counters.limit = limit;
    transient = std::move(candidate);
    return *transient;
  }
  while (!cache.empty() && cache_bytes + candidate->bytes_total > limit) {
    auto victim = std::min_element(
        cache.begin(), cache.end(), [](const auto& lhs, const auto& rhs) {
          return lhs->last_used < rhs->last_used;
        });
    cache_bytes -= (*victim)->bytes_total;
    cache.erase(victim);
    ++counters.evictions;
  }
  cache_bytes += candidate->bytes_total;
  counters.bytes = cache_bytes;
  counters.limit = limit;
  cache.emplace_back(std::move(candidate));
    if (numa_workspace_profile_enabled()) {
      const auto& ws = *cache.back();
    std::cout << "TFS_NUMA_WORKSPACE cache_hit=0,cache_misses="
              << counters.misses << ",cache_hits=" << counters.hits
              << ",cache_evictions=" << counters.evictions
              << ",cache_bytes=" << counters.bytes
              << ",cache_limit_bytes=" << limit
              << ",workspace_total_bytes=" << ws.bytes_total
              << ",local_dw_bytes="
              << (static_cast<std::size_t>(threads) * dp * kp * sizeof(float))
              << ",scratch_bytes=" << ws.scratch_bytes()
              << ",numa_first_touch_ms=" << ws.first_touch_ms << std::endl;
    ws.emit_buffer_profile();
  }
  return *cache.back();
}

// Transform-first High-D native slab.  This is the first native production
// entry point for K>D>128: one D slab is scaled/rounded to BF16, pulled once
// through the CSR graph, and immediately consumed by the AMX dW and dHs
// kernels.  The Python dispatcher may invoke it once per planner slab and
// accumulates dHs in FP32 across slabs.  Keeping the slab boundary at the
// entry point makes the workspace and numerical contract explicit while
// avoiding the historical per-128-column wide-output loop.
std::vector<at::Tensor> c3_backward_transform_highd_amx_v1(
    const at::Tensor& grad_in, const at::Tensor& hs_in,
    const at::Tensor& weight_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, bool compute_dx) {
  auto grad = grad_in.contiguous();
  auto hs = hs_in.contiguous();
  auto weight = weight_in.contiguous();
  auto rowptr = rowptr_in.contiguous();
  auto colidx = colidx_in.contiguous();
  auto scale = scale_in.contiguous();
  TORCH_CHECK(grad.device().is_cpu() && grad.scalar_type() == at::kFloat,
              "transform high-D grad must be CPU FP32");
  TORCH_CHECK(hs.device().is_cpu() && hs.scalar_type() == at::kBFloat16,
              "transform high-D Hs must be CPU BF16");
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == at::kFloat,
              "transform high-D weight must be CPU FP32");
  TORCH_CHECK(scale.device().is_cpu() && scale.scalar_type() == at::kFloat,
              "transform high-D scale must be CPU FP32");
  TORCH_CHECK(rowptr.scalar_type() == at::kLong &&
              colidx.scalar_type() == at::kLong,
              "transform high-D CSR must be int64");
  TORCH_CHECK(grad.dim() == 2 && hs.dim() == 2 && weight.dim() == 2 &&
              rowptr.dim() == 1 && colidx.dim() == 1 && scale.dim() == 1,
              "transform high-D expects rank-2 inputs");
  const int n = static_cast<int>(grad.size(0));
  const int d = static_cast<int>(grad.size(1));
  const int k = static_cast<int>(hs.size(1));
  const int hs_width = static_cast<int>(hs.size(1));
  const int threads = static_cast<int>(threads64);
  TORCH_CHECK(n >= 1 && k > d && d > 128 && threads >= 1 && threads <= 32 &&
              weight.size(0) == k && weight.size(1) == d &&
              hs.size(0) == n && rowptr.numel() == n + 1 &&
              scale.numel() == n,
              "transform high-D shape unsupported");
  const int dp = round_up(d, 32);
  const int kp = round_up(k, 64);
  const int np = round_up(n, 32);
  const int panel = [] {
    const char* value = std::getenv("TFS_HIGHD_NATIVE_PANEL");
    if (value == nullptr || *value == '\0') return 512;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && parsed >= 32,
                "TFS_HIGHD_NATIVE_PANEL must be >=32");
    return round_up(static_cast<int>(parsed), 32);
  }();

  const std::int64_t* rp = rowptr.data_ptr<std::int64_t>();
  const std::int64_t* ci = colidx.data_ptr<std::int64_t>();
  bool e9_index_reused = false;
  const bool use_e9 = formal_colidx_enabled(
      experiment_flag("TFS_GLUE_E9_INT32_COLIDX"), ci, colidx.numel());
  const std::int32_t* ci32 = use_e9
      ? int32_colidx_workspace(ci, colidx.numel(), e9_index_reused) : nullptr;

  std::unique_lock<std::mutex> runtime_lock(highd_stream_runtime_mutex());
  auto& ws = highd_stream_workspace(dp, kp, threads, panel, compute_dx);
  std::fill(ws.local.data(), ws.local.data() + ws.local.size(), 0.0f);

  // Materialize only the current D slab of Gs.  This is deliberately the
  // materialized variant of the T2 gate; a future fused-source-scale probe
  // can share the same entry point without changing the dW/dHs contract.
  auto gs = at::empty({n, dp}, grad.options().dtype(at::kBFloat16));
  const float* gp = grad.data_ptr<float>();
  const float* sp = scale.data_ptr<float>();
  bf16* gsp = reinterpret_cast<bf16*>(gs.data_ptr<at::BFloat16>());
  parallel_range(n, threads, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const float* gr = gp + static_cast<std::size_t>(row) * d;
      bf16* dst = gsp + static_cast<std::size_t>(row) * dp;
      const __m512 sv = _mm512_set1_ps(sp[row]);
      int q = 0;
      for (; q + 16 <= d; q += 16) {
        const __m512 v = _mm512_mul_ps(
            _mm512_loadu_ps(gr + q), sv);
        store_fp32_tail_bf16(dst + q, v, 16);
      }
      for (; q < d; ++q) dst[q] = to_bf16(gr[q] * sp[row]);
      for (; q < dp; ++q) dst[q] = bf16(0);
    }
  });

  // Pack W^T once for dHs.  The padded D tail is explicitly zeroed before
  // packing, so AMX never reads an uninitialized BF16 tile.
  if (compute_dx) {
    std::fill(ws.wt.data(), ws.wt.data() + ws.wt.size(), bf16(0));
    const float* wp = weight.data_ptr<float>();
    parallel_range(dp, threads, [&](std::int64_t begin, std::int64_t end) {
      for (int q = static_cast<int>(begin); q < static_cast<int>(end); ++q) {
        if (q >= d) continue;
        for (int p = 0; p < k; ++p)
          ws.wt.data()[static_cast<std::size_t>(q) * kp + p] =
              to_bf16(wp[static_cast<std::size_t>(p) * d + q]);
      }
    });
    pack_rhs_into(ws.wt.data(), dp, kp, ws.packed_wt.data());
  }

  auto dx_padded = compute_dx
      ? at::empty({np, kp}, grad.options())
      : at::empty({0}, grad.options());
  const bool glue_e2 = experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bf16* hsp = reinterpret_cast<const bf16*>(
      hs.data_ptr<at::BFloat16>());
  const int hs_stride = static_cast<int>(hs.stride(0));
  const auto own = panel_schedule(rp, n, panel, threads);
  parallel_workers(threads, [&](int tid) {
    bv2::configure_amx_tiles_16x64();
    Scratch& z = *ws.scratch[static_cast<std::size_t>(tid)];
    float* local = ws.local.data() +
        static_cast<std::size_t>(tid) * dp * kp;
    for (int panel_id : own[static_cast<std::size_t>(tid)]) {
      const int row0 = panel_id * panel;
      const int valid = std::min(panel, n - row0);
      const int rows = round_up(valid, 32);
      // pull_panel writes only the logical D columns for a tail slab.  Clear
      // the panel first so transpose/dH never consumes stale padded columns.
      std::fill(z.y.data(), z.y.data() +
          static_cast<std::size_t>(rows) * dp, bf16(0));
      pull_panel(rp, ci, ci32, use_e9, gsp, d, dp, row0, valid,
                 z.y.data(), dp, glue_e2, false, nullptr, true);
      if (compute_dx) {
        bv2::dh_amx_4c2a2b(
            z.y.data(), rows, valid, dp, ws.packed_wt.data(), k, kp,
            dx_padded.data_ptr<float>(), row0, nullptr, true, nullptr);
      }
      bv2::transpose_y_avx512_t2(z.y.data(), rows, dp, z.yt.data());
      if (hs_width == kp)
        bv2::pack_h_panel_direct(
            hsp, hs_stride, row0, valid, rows, kp, z.hp.data());
      else
        bv2::pack_h_panel_direct_tail(
            hsp, hs_stride, k, row0, valid, rows, kp, z.hp.data());
      bv2::dw_amx_4c2a2b(
          z.yt.data(), dp, rows, z.hp.data(), k, kp, local, true, nullptr);
    }
    _tile_release();
  });

  // Hs already contains the source-side S scaling.  The backward pull above
  // produces A^T dT; applying S here is the source-side chain rule and is the
  // same final multiply used by the established Python/native C3 paths.
  if (compute_dx) {
    float* dxp = dx_padded.data_ptr<float>();
    parallel_range(n, threads, [&](std::int64_t begin, std::int64_t end) {
      for (std::int64_t row = begin; row < end; ++row) {
        float* dr = dxp + static_cast<std::size_t>(row) * kp;
        const __m512 sv = _mm512_set1_ps(sp[row]);
        int q = 0;
        for (; q + 16 <= k; q += 16)
          _mm512_storeu_ps(dr + q,
              _mm512_mul_ps(_mm512_loadu_ps(dr + q), sv));
        for (; q < k; ++q) dr[q] *= sp[row];
      }
    });
  }

  reduce_dwt_persistent(ws.local.data(), threads, runtime_per_numa(threads),
                        dp * kp, ws.group_reduce.data(),
                        ws.numa_reduce.data(), ws.dwt.data());
  auto dw = at::empty({k, d}, grad.options());
  float* dwp = dw.data_ptr<float>();
  for (int a = 0; a < k; ++a)
    for (int b = 0; b < d; ++b)
      dwp[static_cast<std::size_t>(a) * d + b] =
          ws.dwt.data()[static_cast<std::size_t>(b) * kp + a];
  auto db = grad.sum(0).contiguous();
  at::Tensor dx;
  if (compute_dx)
    dx = dx_padded.narrow(0, 0, n).narrow(1, 0, k).contiguous();
  else
    dx = at::empty({0}, grad.options());
  auto meta = at::tensor(
      std::vector<int64_t>{threads64, panel, dp, kp, e9_index_reused ? 1 : 0},
      at::TensorOptions().dtype(at::kLong));
  if (internal_profile_enabled())
    std::cout << "TFS_TRANSFORM_HIGHD_NATIVE N=" << n << ",K=" << k
              << ",D=" << d << ",threads=" << threads
              << ",panel=" << panel << ",dp=" << dp
              << ",kp=" << kp << ",compute_dx=" << (compute_dx ? 1 : 0)
              << ",e9_index_reused=" << (e9_index_reused ? 1 : 0)
              << std::endl;
  return {dx, dw, db, meta};
}

// Transform-first High-D single-scan candidate.  Unlike the slab adapter
// above, this entry does not materialize a full [N,D] BF16 gradient and does
// not re-enter CSR for every D slab.  Each row panel is pulled once across
// the complete logical D width; the resulting BF16 panel is consumed
// immediately by dHs/dW before the next panel is visited.  Source scaling and
// BF16 rounding are fused into the sparse traversal by
// pull_panel_scaled_grad().  This is intentionally a separate symbol: it is
// a correctness/performance candidate, not a silent replacement for the
// established authority path.
std::vector<at::Tensor> c3_backward_transform_highd_single_scan_amx_v1(
    const at::Tensor& grad_in, const at::Tensor& hs_in,
    const at::Tensor& weight_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, bool compute_dx) {
  auto grad = grad_in.contiguous();
  auto hs = hs_in.contiguous();
  auto weight = weight_in.contiguous();
  auto rowptr = rowptr_in.contiguous();
  auto colidx = colidx_in.contiguous();
  auto scale = scale_in.contiguous();
  TORCH_CHECK(grad.device().is_cpu() && grad.scalar_type() == at::kFloat,
              "transform single-scan grad must be CPU FP32");
  TORCH_CHECK(hs.device().is_cpu() && hs.scalar_type() == at::kBFloat16,
              "transform single-scan Hs must be CPU BF16");
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == at::kFloat,
              "transform single-scan weight must be CPU FP32");
  TORCH_CHECK(scale.device().is_cpu() && scale.scalar_type() == at::kFloat,
              "transform single-scan scale must be CPU FP32");
  TORCH_CHECK(rowptr.scalar_type() == at::kLong &&
              colidx.scalar_type() == at::kLong,
              "transform single-scan CSR must be int64");
  TORCH_CHECK(grad.dim() == 2 && hs.dim() == 2 && weight.dim() == 2 &&
              rowptr.dim() == 1 && colidx.dim() == 1 && scale.dim() == 1,
              "transform single-scan expects rank-2 inputs");
  const int n = static_cast<int>(grad.size(0));
  const int d = static_cast<int>(grad.size(1));
  const int k = static_cast<int>(hs.size(1));
  const int threads = static_cast<int>(threads64);
  TORCH_CHECK(n >= 1 && k > d && d > 128 && threads >= 1 && threads <= 32 &&
              weight.size(0) == k && weight.size(1) == d && hs.size(0) == n &&
              rowptr.numel() == n + 1 && scale.numel() == n,
              "transform single-scan shape unsupported");
  const int dp = round_up(d, 32);
  const int kp = round_up(k, 64);
  const int np = round_up(n, 32);
  const int panel = [] {
    const char* value = std::getenv("TFS_HIGHD_NATIVE_PANEL");
    if (value == nullptr || *value == '\0') return 512;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && parsed >= 32,
                "TFS_HIGHD_NATIVE_PANEL must be >=32");
    return round_up(static_cast<int>(parsed), 32);
  }();

  const std::int64_t* rp = rowptr.data_ptr<std::int64_t>();
  const std::int64_t* ci = colidx.data_ptr<std::int64_t>();
  bool e9_index_reused = false;
  const bool use_e9 = formal_colidx_enabled(
      experiment_flag("TFS_GLUE_E9_INT32_COLIDX"), ci, colidx.numel());
  const std::int32_t* ci32 = use_e9
      ? int32_colidx_workspace(ci, colidx.numel(), e9_index_reused) : nullptr;

  std::unique_lock<std::mutex> runtime_lock(highd_stream_runtime_mutex());
  auto& ws = highd_stream_workspace(dp, kp, threads, panel, compute_dx);
  std::fill(ws.local.data(), ws.local.data() + ws.local.size(), 0.0f);

  // Pack W^T once for the complete D width.  No full [N,D] BF16 Gs buffer is
  // allocated; each worker owns only its current row-panel scratch.
  if (compute_dx) {
    std::fill(ws.wt.data(), ws.wt.data() + ws.wt.size(), bf16(0));
    const float* wp = weight.data_ptr<float>();
    parallel_range(dp, threads, [&](std::int64_t begin,std::int64_t end) {
      for (int q = static_cast<int>(begin); q < static_cast<int>(end); ++q) {
        if (q >= d) continue;
        for (int p = 0; p < k; ++p)
          ws.wt.data()[static_cast<std::size_t>(q) * kp + p] =
              to_bf16(wp[static_cast<std::size_t>(p) * d + q]);
      }
    });
    pack_rhs_into(ws.wt.data(), dp, kp, ws.packed_wt.data());
  }

  auto dx_padded = compute_dx
      ? at::empty({np, kp}, grad.options())
      : at::empty({0}, grad.options());
  const bool glue_e2 = experiment_flag("TFS_GLUE_E2_VEC_STORE");
  const bf16* hsp = reinterpret_cast<const bf16*>(
      hs.data_ptr<at::BFloat16>());
  const int hs_stride = static_cast<int>(hs.stride(0));
  const float* gp = grad.data_ptr<float>();
  const float* sp = scale.data_ptr<float>();
  const auto own = panel_schedule(rp, n, panel, threads);
  parallel_workers(threads, [&](int tid) {
    bv2::configure_amx_tiles_16x64();
    Scratch& z = *ws.scratch[static_cast<std::size_t>(tid)];
    float* local = ws.local.data() +
        static_cast<std::size_t>(tid) * dp * kp;
    for (int panel_id : own[static_cast<std::size_t>(tid)]) {
      const int row0 = panel_id * panel;
      const int valid = std::min(panel, n - row0);
      const int rows = round_up(valid, 32);
      std::fill(z.y.data(), z.y.data() +
          static_cast<std::size_t>(rows) * dp, bf16(0));
      // One CSR traversal for the complete D width, followed immediately by
      // dense AMX consumers.  There is no per-slab pull or full Gs tensor.
      pull_panel_scaled_grad(rp, ci, ci32, use_e9, gp, sp, d, d, row0,
                             valid, z.y.data(), dp);
      if (compute_dx) {
        bv2::dh_amx_4c2a2b(
            z.y.data(), rows, valid, dp, ws.packed_wt.data(), k, kp,
            dx_padded.data_ptr<float>(), row0, nullptr, true, nullptr);
      }
      bv2::transpose_y_avx512_t2(z.y.data(), rows, dp, z.yt.data());
      if (static_cast<int>(hs.stride(0)) == kp)
        bv2::pack_h_panel_direct(
            hsp, hs_stride, row0, valid, rows, kp, z.hp.data());
      else
        bv2::pack_h_panel_direct_tail(
            hsp, hs_stride, k, row0, valid, rows, kp, z.hp.data());
      bv2::dw_amx_4c2a2b(
          z.yt.data(), dp, rows, z.hp.data(), k, kp, local, true, nullptr);
    }
    _tile_release();
  });

  if (compute_dx) {
    float* dxp = dx_padded.data_ptr<float>();
    parallel_range(n, threads, [&](std::int64_t begin, std::int64_t end) {
      for (std::int64_t row = begin; row < end; ++row) {
        float* dr = dxp + static_cast<std::size_t>(row) * kp;
        const __m512 sv = _mm512_set1_ps(sp[row]);
        int q = 0;
        for (; q + 16 <= k; q += 16)
          _mm512_storeu_ps(dr + q,
              _mm512_mul_ps(_mm512_loadu_ps(dr + q), sv));
        for (; q < k; ++q) dr[q] *= sp[row];
      }
    });
  }

  reduce_dwt_persistent(ws.local.data(), threads, runtime_per_numa(threads),
                        dp * kp, ws.group_reduce.data(), ws.numa_reduce.data(),
                        ws.dwt.data());
  auto dw = at::empty({k, d}, grad.options());
  float* dwp = dw.data_ptr<float>();
  for (int a = 0; a < k; ++a)
    for (int b = 0; b < d; ++b)
      dwp[static_cast<std::size_t>(a) * d + b] =
          ws.dwt.data()[static_cast<std::size_t>(b) * kp + a];
  auto db = grad.sum(0).contiguous();
  at::Tensor dx;
  if (compute_dx)
    dx = dx_padded.narrow(0, 0, n).narrow(1, 0, k).contiguous();
  else
    dx = at::empty({0}, grad.options());
  auto meta = at::tensor(
      std::vector<int64_t>{threads64, panel, dp, kp,
                           e9_index_reused ? 1 : 0, 1},
      at::TensorOptions().dtype(at::kLong));
  if (internal_profile_enabled())
    std::cout << "TFS_TRANSFORM_HIGHD_SINGLE_SCAN N=" << n
              << ",K=" << k << ",D=" << d << ",threads=" << threads
              << ",panel=" << panel << ",dp=" << dp << ",kp=" << kp
              << ",compute_dx=" << (compute_dx ? 1 : 0)
              << ",e9_index_reused=" << (e9_index_reused ? 1 : 0)
              << std::endl;
  return {dx, dw, db, meta};
}

std::vector<at::Tensor> c3_backward_aggregate_highd_amx_v1(
    const at::Tensor& grad_in, const at::Tensor& pulled_in,
    const at::Tensor& weight_in, const at::Tensor& rowptr_in,
    const at::Tensor& colidx_in, const at::Tensor& scale_in,
    int64_t threads64, bool compute_dx) {
  auto grad = grad_in.contiguous();
  auto pulled = pulled_in.contiguous();
  auto weight = weight_in.contiguous();
  auto rowptr = rowptr_in.contiguous();
  auto colidx = colidx_in.contiguous();
  auto scale = scale_in.contiguous();
  TORCH_CHECK(grad.device().is_cpu() && grad.scalar_type() == at::kFloat &&
              pulled.device().is_cpu() &&
              pulled.scalar_type() == at::kBFloat16 &&
              weight.device().is_cpu() && weight.scalar_type() == at::kFloat &&
              scale.device().is_cpu() && scale.scalar_type() == at::kFloat,
              "aggregate high-D backward expects CPU FP32/BF16 tensors");
  TORCH_CHECK(rowptr.scalar_type() == at::kLong &&
              colidx.scalar_type() == at::kLong,
              "aggregate high-D backward CSR must be int64");
  TORCH_CHECK(grad.dim() == 2 && pulled.dim() == 2 && weight.dim() == 2 &&
              rowptr.dim() == 1 && colidx.dim() == 1 && scale.dim() == 1,
              "aggregate high-D backward expects rank-2 tensors");
  const int n = static_cast<int>(grad.size(0));
  const int d = static_cast<int>(grad.size(1));
  const int k = static_cast<int>(weight.size(0));
  const int threads = static_cast<int>(threads64);
  TORCH_CHECK(n >= 1 && k >= 1 && d > 128 && threads >= 1 && threads <= 32 &&
              pulled.size(0) == n && pulled.size(1) == k &&
              weight.size(1) == d && rowptr.numel() == n + 1 &&
              scale.numel() == n,
              "aggregate high-D backward shape unsupported");
  // AMX row kernels require complete 32-row tiles.  The logical K tail is
  // supported, but the pull-only primitive can consume a padded temporary
  // only after it is compacted, so retain the logical K tensor for now.
  const int dp = round_up(d, 32);
  const int kp = round_up(k, 64);
  const int np = round_up(n, 32);
  int panel = 512;
  if (const char* value = std::getenv("TFS_HIGHD_NATIVE_PANEL")) {
    char* end = nullptr;
    const long requested = std::strtol(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && requested >= 32,
                "TFS_HIGHD_NATIVE_PANEL must be >=32");
    panel = round_up(static_cast<int>(requested), 32);
  }
  const bool use_k_tail = kp != k;
  std::unique_lock<std::mutex> runtime_lock(highd_stream_runtime_mutex());
  auto& ws = highd_stream_workspace(dp, kp, threads, panel, compute_dx);
  if (numa_workspace_profile_enabled()) {
    const auto& counters = highd_workspace_cache_counters();
    std::cout << "TFS_NUMA_WORKSPACE cache_hit=" << (ws.last_cache_hit ? 1 : 0)
              << ",cache_hits="
              << counters.hits << ",cache_misses=" << counters.misses
              << ",cache_evictions=" << counters.evictions
              << ",cache_bytes=" << counters.bytes
              << ",cache_limit_bytes=" << counters.limit
              << ",workspace_total_bytes=" << ws.bytes_total
              << ",local_dw_bytes="
              << (static_cast<std::size_t>(threads) * dp * kp * sizeof(float))
              << ",scratch_bytes=" << ws.scratch_bytes()
              << ",numa_first_touch_ms=" << ws.first_touch_ms << std::endl;
    ws.emit_buffer_profile();
  }
  const double profile_t0 = now_ms();

  // Convert and pack W^T once.  The padded tail is explicitly zeroed so the
  // AMX tile kernels never read uninitialized BF16 values.
  const double profile_wt0 = now_ms();
  if (compute_dx) {
    std::fill(ws.wt.data(), ws.wt.data() + ws.wt.size(), bf16(0));
    const float* wp = weight.data_ptr<float>();
    parallel_range(dp, threads, [&](std::int64_t begin,std::int64_t end) {
      for (int q = static_cast<int>(begin); q < static_cast<int>(end); ++q)
        if (q < d)
          for (int p = 0; p < k; ++p)
            ws.wt.data()[static_cast<std::size_t>(q) * kp + p] =
                to_bf16(wp[static_cast<std::size_t>(p) * d + q]);
    });
    pack_rhs_into(ws.wt.data(), dp, kp, ws.packed_wt.data());
  }
  const double profile_wt1 = now_ms();

  auto dw = at::empty({k, d}, grad.options());
  // Bias gradient is accumulated while the panel gradient is already being
  // read for scale+BF16 conversion.  The previous ``grad.sum(0)`` launched a
  // second full N*D pass (about 12 GB of FP32 traffic for IGB-2983), which
  // showed up as a ~100 ms fixed cost in the native high-D backward.
  auto db = at::empty({d}, grad.options());
  at::Tensor d_padded;
  // dh_amx_4c2a2b stores complete 32-row AMX tiles.  Allocate the row tail
  // explicitly and narrow it back before the CSR pull; otherwise an N that is
  // not a multiple of 32 writes past the logical dX tensor.
  if (compute_dx) d_padded = at::empty({np, kp}, grad.options());
  const double profile_alloc1 = now_ms();
  const float* gp = grad.data_ptr<float>();
  const float* sp = scale.data_ptr<float>();
  const bf16* pp = reinterpret_cast<const bf16*>(
      pulled.data_ptr<at::BFloat16>());
  const int panel_count = (n + panel - 1) / panel;
  const bool profile_native = profile_clock_enabled();
  // Dense work is uniform across rows.  For sufficiently many panels, a
  // contiguous ownership range improves NUMA locality and keeps the scale /
  // transpose passes in the same worker's cache.  Tiny fixtures with fewer
  // than four panels per worker retain the cyclic schedule to avoid idle
  // workers.  Both decisions remain explicitly overridable for ablations.
  const bool large_panel_work = panel_count >= threads * 4;
  // The fused db update adds a read/modify/write of the per-thread D-vector
  // to every scale row.  That is useful for narrow outputs, but for wide
  // outputs the vectorized standalone grad.sum(0) is cheaper despite the
  // second pass.  Keep the default dimension-driven and retain the env knob
  // for explicit ablations (or future hardware-specific tuning).
  const bool fused_db = highd_adaptive_flag("TFS_HIGHD_FUSED_DB", d <= 512);
  const bool fused_transpose = highd_adaptive_flag(
      "TFS_HIGHD_FUSED_TRANSPOSE", large_panel_work);
  const bool fused_scale_transpose = fused_transpose && highd_adaptive_flag(
      "TFS_HIGHD_FUSED_SCALE_TRANSPOSE", false);
  const bool contiguous_panels = highd_adaptive_flag(
      "TFS_HIGHD_CONTIGUOUS_PANELS", large_panel_work);
  std::vector<double> native_scale_ms(threads), native_dh_ms(threads),
      native_transpose_ms(threads), native_pack_ms(threads),
      native_dw_ms(threads), native_fused_scale_transpose_ms(threads),
      native_thread_ms(threads);

  const double profile_kernel0 = now_ms();
  parallel_workers(threads, [&](int tid) {
    const double thread_t0 = profile_native ? now_ms() : 0.0;
    float* dbp = ws.db_local.data() +
        static_cast<std::size_t>(tid) * dp;
    std::fill(dbp, dbp + dp, 0.0f);
    float* local = ws.local.data() +
        static_cast<std::size_t>(tid) * dp * kp;
    std::fill(local, local + static_cast<std::size_t>(dp) * kp, 0.0f);
    bv2::configure_amx_tiles_16x64();
    Scratch& z = *ws.scratch[static_cast<std::size_t>(tid)];
    const int panel_begin = contiguous_panels
        ? panel_count * tid / threads : tid;
    const int panel_end = contiguous_panels
        ? panel_count * (tid + 1) / threads : panel_count;
    for (int panel_id = panel_begin;
         contiguous_panels ? panel_id < panel_end
                            : panel_id < panel_count;
         contiguous_panels ? ++panel_id : panel_id += threads) {
      const int row0 = panel_id * panel;
      const int valid = std::min(panel, n - row0);
      const int rows = round_up(valid, 32);
      auto scale_rows = [&](int begin_row, int end_row) {
        for (int row = begin_row; row < end_row; ++row) {
          const float* gr = gp + static_cast<std::size_t>(row0 + row) * d;
          bf16* yr = z.y.data() + static_cast<std::size_t>(row) * dp;
          const __m512 sv = _mm512_set1_ps(sp[row0 + row]);
          int q = 0;
          for (; q + 32 <= d; q += 32) {
            const __m512 g0 = _mm512_loadu_ps(gr + q);
            const __m512 g1 = _mm512_loadu_ps(gr + q + 16);
            if (fused_db) {
              const __m512 b0 = _mm512_add_ps(
                  _mm512_loadu_ps(dbp + q), g0);
              const __m512 b1 = _mm512_add_ps(
                  _mm512_loadu_ps(dbp + q + 16), g1);
              _mm512_storeu_ps(dbp + q, b0);
              _mm512_storeu_ps(dbp + q + 16, b1);
            }
            const __m512 v0 = _mm512_mul_ps(g0, sv);
            const __m512 v1 = _mm512_mul_ps(g1, sv);
            _mm512_storeu_si512(reinterpret_cast<void*>(yr + q),
                (__m512i)_mm512_cvtne2ps_pbh(v1, v0));
          }
          for (; q < d; ++q) {
            if (fused_db) dbp[q] += gr[q];
            yr[q] = to_bf16(gr[q] * sp[row0 + row]);
          }
          for (; q < dp; ++q) yr[q] = bf16(0);
        }
      };
      const double scale_t0 = profile_native ? now_ms() : 0.0;
      if (fused_transpose) {
        // T3 transposes each freshly produced 32-row block while it is still
        // hot in cache, avoiding a second pass over the complete panel.
        for (int block = 0; block < rows; block += 32) {
          const int block_end = std::min(valid, block + 32);
          if (fused_scale_transpose) {
            const double fused_t0 = profile_native ? now_ms() : 0.0;
            bv2::scale_bf16_db_transpose_t3(
                gp, sp, row0 + block, block_end - block, d, dp, rows, block,
                z.y.data() + static_cast<std::size_t>(block) * dp,
                z.yt.data(), fused_db ? dbp : nullptr);
            if (profile_native) {
              const double fused_elapsed = now_ms() - fused_t0;
              native_fused_scale_transpose_ms[tid] += fused_elapsed;
              native_scale_ms[tid] += fused_elapsed;
            }
          } else {
            scale_rows(block, block_end);
            if (block_end < block + 32)
              std::fill(z.y.data() + static_cast<std::size_t>(block_end) * dp,
                        z.y.data() + static_cast<std::size_t>(block + 32) * dp,
                        bf16(0));
            const double transpose_t0 = profile_native ? now_ms() : 0.0;
            bv2::transpose_y_block_to_panel_t3(
                z.y.data() + static_cast<std::size_t>(block) * dp, 32, dp,
                z.yt.data(), rows, block);
            if (profile_native)
              native_transpose_ms[tid] += now_ms() - transpose_t0;
          }
        }
      } else {
        scale_rows(0, valid);
        if (rows > valid)
          std::fill(z.y.data() + static_cast<std::size_t>(valid) * dp,
                    z.y.data() + static_cast<std::size_t>(rows) * dp, bf16(0));
      }
      if (profile_native) native_scale_ms[tid] += now_ms() - scale_t0;

      double stage_t0 = profile_native ? now_ms() : 0.0;
      if (compute_dx)
        bv2::dh_amx_4c2a2b(z.y.data(), rows, valid, dp,
                           ws.packed_wt.data(), k, kp,
                           d_padded.data_ptr<float>(), row0, nullptr, true,
                           nullptr);
      if (profile_native) native_dh_ms[tid] += now_ms() - stage_t0;
      if (!fused_transpose) {
        stage_t0 = profile_native ? now_ms() : 0.0;
        bv2::transpose_y_avx512_t2(z.y.data(), rows, dp, z.yt.data());
        if (profile_native) native_transpose_ms[tid] += now_ms() - stage_t0;
      }
      stage_t0 = profile_native ? now_ms() : 0.0;
      if (use_k_tail)
        bv2::pack_h_panel_direct_tail(pp, k, k, row0, valid, rows, kp,
                                      z.hp.data());
      else
        bv2::pack_h_panel_direct(pp, k, row0, valid, rows, kp, z.hp.data());
      if (profile_native) native_pack_ms[tid] += now_ms() - stage_t0;
      stage_t0 = profile_native ? now_ms() : 0.0;
      bv2::dw_amx_4c2a2b(z.yt.data(), dp, rows, z.hp.data(), k, kp, local,
                         true, nullptr);
      if (profile_native) native_dw_ms[tid] += now_ms() - stage_t0;
    }
    if (profile_native) native_thread_ms[tid] = now_ms() - thread_t0;
    _tile_release();
  });
  const double profile_kernel1 = now_ms();

  // Keep the bias-gradient reduction inside the reported critical path for
  // both ablation arms.  In particular, the non-fused arm intentionally uses
  // the old grad.sum(0) implementation; excluding it would make the profile
  // comparison systematically favor that arm.
  const double profile_db0 = now_ms();
  if (fused_db) {
    float* db_out = db.data_ptr<float>();
    for (int q = 0; q < d; ++q) {
      float sum = 0.0f;
      for (int tid = 0; tid < threads; ++tid)
        sum += ws.db_local.data()[static_cast<std::size_t>(tid) * dp + q];
      db_out[q] = sum;
    }
  } else {
    // Keep an explicit ablation against the framework reduction.  This is
    // intentionally outside the panel loop so the fused path can be measured
    // against the exact old ``grad.sum(0)`` semantics.
    db.copy_(grad.sum(0));
  }
  const double profile_db1 = now_ms();

  const double profile_reduce0 = profile_db1;
  const bool parallel_reduce =
      experiment_flag("TFS_HIGHD_PARALLEL_REDUCE") && threads > 1;
  if (parallel_reduce) {
    // The legacy hierarchical reducer is intentionally deterministic but
    // serial over the whole Kp*Dp result.  High-D dW makes that reduction a
    // visible tail (about 7 ms even in the small synthetic gate).  Reuse the
    // persistent worker pool to split output elements while retaining the
    // exact group/NUMA summation order per element.
    reduce_dwt_parallel_deterministic(
        ws.local.data(), threads, runtime_per_numa(threads), dp * kp,
        ws.dwt.data());
  } else {
    reduce_dwt_persistent(ws.local.data(), threads, runtime_per_numa(threads),
                          dp * kp, ws.group_reduce.data(),
                          ws.numa_reduce.data(), ws.dwt.data());
  }
  const double profile_reduce1 = now_ms();
  float* dwp = dw.data_ptr<float>();
  const float* dwt = ws.dwt.data();
  for (int a = 0; a < k; ++a)
    for (int b = 0; b < d; ++b)
      dwp[static_cast<std::size_t>(a) * d + b] =
          dwt[static_cast<std::size_t>(b) * kp + a];
  const double profile_scatter = now_ms();

  at::Tensor dx;
  const double profile_pull0 = now_ms();
  if (compute_dx) {
    at::Tensor pull_input = d_padded.narrow(0, 0, n);
    pull_input = use_k_tail
        ? pull_input.narrow(1, 0, k).contiguous() : pull_input;
    auto d_h = c3_pull_only_amx_v1(pull_input, rowptr, colidx, threads);
    // c3_pull_only_amx_v1 returns a newly allocated FP32 tensor.  Scale it
    // in place instead of forming a second N x K tensor through the
    // out-of-place broadcast multiply; this removes one full read/write of
    // the high-D backward dX buffer and keeps the result mathematically
    // identical to the previous expression.
    d_h.mul_(scale.view({n, 1}));
    dx = d_h;
  } else {
    dx = at::empty({0}, grad.options());
  }
  const double profile_pull1 = now_ms();
  auto meta = at::tensor({threads, panel, dp, kp},
                         at::TensorOptions().dtype(at::kLong));
  if (internal_profile_enabled()) {
    auto maxv = [](const std::vector<double>& values) {
      return *std::max_element(values.begin(), values.end());
    };
    std::cout << "TFS_HIGHD_NATIVE panels=" << panel_count
              << ",panel=" << panel << ",N=" << n << ",K=" << k
              << ",D=" << d << ",threads=" << threads
              << ",compute_dx=" << (compute_dx ? 1 : 0) << std::endl;
    std::cout << std::fixed << std::setprecision(6)
              << "TFS_HIGHD_NATIVE_PROFILE N=" << n << ",K=" << k
              << ",D=" << d << ",threads=" << threads
              << ",panel=" << panel << ",wt_pack_ms="
              << (profile_wt1 - profile_wt0)
              << ",alloc_db_ms=" << (profile_alloc1 - profile_wt1)
              << ",kernel_ms=" << (profile_kernel1 - profile_kernel0)
              << ",parallel_reduce=" << (parallel_reduce ? 1 : 0)
              << ",fused_db=" << (fused_db ? 1 : 0)
              << ",fused_transpose=" << (fused_transpose ? 1 : 0)
              << ",fused_scale_transpose="
              << (fused_scale_transpose ? 1 : 0)
              << ",contiguous_panels=" << (contiguous_panels ? 1 : 0)
              << ",db_reduce_ms=" << (profile_db1 - profile_db0)
              << ",reduce_ms=" << (profile_reduce1 - profile_reduce0)
              << ",scatter_ms=" << (profile_scatter - profile_reduce1)
              << ",pull_scale_ms=" << (profile_pull1 - profile_pull0)
              << ",scale_kernel_ms=" << maxv(native_scale_ms)
              << ",dh_kernel_ms=" << maxv(native_dh_ms)
              << ",transpose_ms=" << maxv(native_transpose_ms)
              << ",fused_scale_transpose_ms="
              << maxv(native_fused_scale_transpose_ms)
              << ",pack_p_ms=" << maxv(native_pack_ms)
              << ",dw_kernel_ms=" << maxv(native_dw_ms)
              << ",thread_ms=" << maxv(native_thread_ms)
              << ",total_ms=" << (profile_pull1 - profile_t0)
              << std::endl;
  }
  return {dx, dw, db, meta};
}

std::vector<at::Tensor> c3_backward_saved_t_amx_v3(
    const at::Tensor& grad_in, const at::Tensor& saved_t_in,
    const at::Tensor& rowptr_in, const at::Tensor& colidx_in,
    const at::Tensor& scale_in,
    int64_t threads64) {
  const double profile_t0=now_ms();
  TORCH_CHECK(grad_in.device().is_cpu() && grad_in.scalar_type()==at::kFloat,
              "saved-T backward grad must be CPU FP32");
  TORCH_CHECK(saved_t_in.scalar_type()==at::kBFloat16,
              "saved-T backward expects BF16 T");
  TORCH_CHECK(scale_in.scalar_type()==at::kFloat,
              "saved-T backward scale must be FP32");
  TORCH_CHECK(rowptr_in.scalar_type()==at::kLong &&
              colidx_in.scalar_type()==at::kLong,
              "saved-T backward CSR must be int64");
  auto grad=grad_in.contiguous(),saved_t=saved_t_in.contiguous();
  auto rp_t=rowptr_in.contiguous(),ci_t=colidx_in.contiguous();
  auto scale=scale_in.contiguous();
  const int n=static_cast<int>(grad.size(0));
  const int d=static_cast<int>(grad.size(1));
  const int k=static_cast<int>(saved_t.size(1));
  const int threads=static_cast<int>(threads64);
  const int dp=round_up(d,32),kp=round_up(k,64),panel=512;
  TORCH_CHECK(saved_t.size(0)==n && scale.numel()==n && rp_t.numel()==n+1 &&
              k<=128 && d<=128 && threads>=1 && threads<=32,
              "saved-T backward shape unsupported");
  const float* gp=grad.data_ptr<float>();
  const float* sp=scale.data_ptr<float>();
  const bf16* tp=reinterpret_cast<const bf16*>(
      saved_t.data_ptr<at::BFloat16>());
  const std::int64_t* rp=rp_t.data_ptr<std::int64_t>();
  const std::int64_t* ci=ci_t.data_ptr<std::int64_t>();
  std::unique_lock<std::mutex> workspace_lock(backward_workspace_mutex());
  bool workspace_reused=false;
  BackwardWorkspaceSpec workspace_spec{
      n,d,k,dp,kp,threads,panel,rp,ci,false,true,false,false,true,false,
      BackwardImplKind::aggregate_static_v3};
  bool workspace_transient=false;
  BackwardWorkspace& ws=backward_workspace(workspace_spec,workspace_reused,workspace_transient);
  const double profile_stage0=now_ms();
  parallel_workers(threads,[&](int tid){
    const std::int64_t begin=static_cast<std::int64_t>(n)*tid/threads;
    const std::int64_t end=static_cast<std::int64_t>(n)*(tid+1)/threads;
    float* dbp=ws.db_local.data()+static_cast<std::size_t>(tid)*dp;
    std::fill(dbp,dbp+dp,0.0f);
    for(std::int64_t i=begin;i<end;++i){
      bf16* dst=ws.gs.data()+static_cast<std::size_t>(i)*dp;
      const float* src=gp+static_cast<std::size_t>(i)*d;
      for(int q=0;q<d;++q){
        dbp[q]+=src[q];
        dst[q]=to_bf16(src[q]*sp[i]);
      }
    }
  });
  const double profile_stage1=now_ms();
  auto dw=at::empty({k,d},grad.options());
  auto db=at::empty({d},grad.options());
  float* db_out=db.data_ptr<float>();
  for(int q=0;q<d;++q){
    float sum=0.0f;
    for(int tid=0;tid<threads;++tid)
      sum+=ws.db_local.data()[static_cast<std::size_t>(tid)*dp+q];
    db_out[q]=sum;
  }
  const int panel_count=(n+panel-1)/panel;
  std::vector<double> transpose_ms(threads),pack_t_ms(threads),dw_ms(threads);
  const double profile_kernel0=now_ms();
  parallel_workers(threads,[&](int tid){
    float* slab=ws.local.data()+static_cast<std::size_t>(tid)*dp*kp;
    std::fill(slab,slab+static_cast<std::size_t>(dp)*kp,0.0f);
    bv2::configure_amx_tiles_16x64();
    Scratch& z=*ws.scratch[tid];
    for(int p=tid;p<panel_count;p+=threads){
      const int row0=p*panel,valid=std::min(panel,n-row0);
      const int rows=round_up(valid,32);
      const bf16* grad_panel=ws.gs.data()+static_cast<std::size_t>(row0)*dp;
      if(rows>valid){
        std::memcpy(z.y.data(),grad_panel,
                    static_cast<std::size_t>(valid)*dp*sizeof(bf16));
        std::fill(z.y.data()+static_cast<std::size_t>(valid)*dp,
                  z.y.data()+static_cast<std::size_t>(rows)*dp,bf16(0));
        grad_panel=z.y.data();
      }
      double q=now_ms();
      bv2::transpose_y_avx512_t2(grad_panel,rows,dp,z.yt.data());
      transpose_ms[tid]+=now_ms()-q;q=now_ms();
      bv2::pack_h_panel_direct_tail(tp,k,k,row0,valid,rows,kp,z.hp.data());
      pack_t_ms[tid]+=now_ms()-q;q=now_ms();
      bv2::dw_amx_4c2a2b(z.yt.data(),dp,rows,z.hp.data(),k,kp,slab,
                          true,nullptr);
      dw_ms[tid]+=now_ms()-q;
    }
    _tile_release();
  });
  const double profile_kernel1=now_ms();
  reduce_dwt_persistent(ws.local.data(),threads,runtime_per_numa(threads),dp*kp,
                        ws.group_reduce.data(),ws.numa_reduce.data(),
                        ws.dwt.data());
  const double profile_reduce=now_ms();
  float* dwp=dw.data_ptr<float>();
  for(int a=0;a<k;++a)for(int b=0;b<d;++b)
    dwp[static_cast<std::size_t>(a)*d+b]=
        ws.dwt.data()[static_cast<std::size_t>(b)*kp+a];
  const double profile_end=now_ms();
  if(internal_profile_enabled()){
    auto maxv=[](const std::vector<double>& v){
      return *std::max_element(v.begin(),v.end());};
    std::cout<<"TFS_AGG_SAVED compute_dx=0 pulled_bytes="
      <<(saved_t.numel()*saved_t.element_size())
      <<",backward_sparse_calls=0,dw_ms="<<maxv(dw_ms)<<std::endl;
    std::cout<<std::fixed<<std::setprecision(6)
      <<"TFS_INTERNAL,kind=backward_saved_t,k="<<k<<",d="<<d
      <<",threads="<<threads<<",workspace_reused="<<(workspace_reused?1:0)
      <<",workspace_transient="<<(workspace_transient?1:0)
      <<",workspace_cache_limit_bytes="<<workspace_cache_limit_bytes()
      <<",stage_grad_db_ms="<<(profile_stage1-profile_stage0)
      <<",transpose_thread_max_ms="<<maxv(transpose_ms)
      <<",pack_t_thread_max_ms="<<maxv(pack_t_ms)
      <<",dw_thread_max_ms="<<maxv(dw_ms)
      <<",kernel_wall_ms="<<(profile_kernel1-profile_kernel0)
      <<",reduce_ms="<<(profile_reduce-profile_kernel1)
      <<",scatter_ms="<<(profile_end-profile_reduce)
      <<",total_ms="<<(profile_end-profile_t0)<<std::endl;
  }
  auto meta=at::tensor({threads,512,dp,kp},
      at::TensorOptions().dtype(at::kLong));
  return {dw,db,meta};
}
