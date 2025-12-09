/**
 * Micro-benchmark: SIMD Inner Product Loop vs SGEMM
 *
 * Scenario: 1 query vector vs 2048 data vectors
 * Compare:
 *   1. hnswlib SIMD (loop over vectors, function call per vector)
 *   2. SGEMM (batch matrix multiplication via BLAS)
 *   3. Batched SIMD (inline, no function call overhead)
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cstring>

// BLAS header
#ifdef USE_BLAS
extern "C" {
    void cblas_sgemv(int order, int trans, int m, int n,
                     float alpha, const float *a, int lda,
                     const float *x, int incx,
                     float beta, float *y, int incy);

    void cblas_sgemm(int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha, const float *A, int lda,
                     const float *B, int ldb,
                     float beta, float *C, int ldc);

    // CBLAS constants
    #define CblasRowMajor 101
    #define CblasColMajor 102
    #define CblasNoTrans 111
    #define CblasTrans 112
}
#endif

// hnswlib headers
#include "hnswlib.h"
#include "space_ip.h"

//==============================================================================
// Configuration
//==============================================================================
constexpr size_t NUM_VECTORS = 2048;      // Number of data vectors
constexpr size_t NUM_WARMUP = 100;        // Warmup iterations
constexpr size_t NUM_ITERATIONS = 1000;   // Benchmark iterations
constexpr size_t ALIGNMENT = 64;          // Memory alignment for AVX-512

//==============================================================================
// Utility Functions
//==============================================================================

// Aligned memory allocation
template<typename T>
T* aligned_alloc_array(size_t count) {
    size_t size = ((count * sizeof(T) + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    void* ptr = std::aligned_alloc(ALIGNMENT, size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, size);
    return static_cast<T*>(ptr);
}

// Generate random float data
void generate_random_data(float* data, size_t count, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(gen);
    }
}

// Normalize vector (for inner product = cosine similarity)
void normalize_vector(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
}

// Statistics calculation
struct Stats {
    double mean;
    double std_dev;
    double min_val;
    double max_val;
    double median;
};

Stats calculate_stats(std::vector<double>& times) {
    Stats s;
    std::sort(times.begin(), times.end());

    s.min_val = times.front();
    s.max_val = times.back();
    s.median = times[times.size() / 2];

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    s.mean = sum / times.size();

    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - s.mean) * (t - s.mean);
    }
    s.std_dev = std::sqrt(sq_sum / times.size());

    return s;
}

//==============================================================================
// Benchmark Implementations
//==============================================================================

// Method 1: SIMD loop (hnswlib style) - one function call per vector
void benchmark_simd_loop(
    const float* query,
    const float* data,
    float* results,
    size_t dim,
    size_t num_vectors,
    hnswlib::DISTFUNC<float> dist_func,
    void* dist_param
) {
    for (size_t i = 0; i < num_vectors; ++i) {
        // Inner product distance is 1 - IP, so convert back
        float dist = dist_func(query, data + i * dim, dist_param);
        results[i] = 1.0f - dist;
    }
}

#ifdef USE_BLAS
// Method 2: SGEMM batch - matrix multiplication
// query: 1 x dim (row vector)
// data: num_vectors x dim (each row is a vector)
// result: 1 x num_vectors = query @ data^T
void benchmark_sgemm(
    const float* query,      // 1 x dim
    const float* data,       // num_vectors x dim (row-major)
    float* results,          // 1 x num_vectors
    size_t dim,
    size_t num_vectors
) {
    // C = alpha * A * B^T + beta * C
    // A: 1 x dim (query)
    // B: num_vectors x dim -> B^T: dim x num_vectors
    // C: 1 x num_vectors (results)
    //
    // M = 1, N = num_vectors, K = dim
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,           // A not transposed
        CblasTrans,             // B transposed
        1,                      // M: rows of A
        static_cast<int>(num_vectors),  // N: cols of B^T
        static_cast<int>(dim),  // K: cols of A
        1.0f,                   // alpha
        query,                  // A
        static_cast<int>(dim),  // lda
        data,                   // B
        static_cast<int>(dim),  // ldb
        0.0f,                   // beta
        results,                // C
        static_cast<int>(num_vectors)   // ldc
    );
}

// Method 3: SGEMV - matrix-vector multiplication
// Treating data as a matrix and query as a vector
// result = data @ query (each row of data dot query)
void benchmark_sgemv(
    const float* query,      // dim x 1
    const float* data,       // num_vectors x dim (row-major)
    float* results,          // num_vectors x 1
    size_t dim,
    size_t num_vectors
) {
    // y = alpha * A * x + beta * y
    // A: num_vectors x dim
    // x: dim x 1
    // y: num_vectors x 1
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,           // A not transposed
        static_cast<int>(num_vectors),  // M: rows of A
        static_cast<int>(dim),  // N: cols of A
        1.0f,                   // alpha
        data,                   // A
        static_cast<int>(dim),  // lda
        query,                  // x
        1,                      // incx
        0.0f,                   // beta
        results,                // y
        1                       // incy
    );
}
#endif

// Method 4: Batched SIMD - inline computation without function call overhead
#if defined(USE_AVX)
void benchmark_batched_avx(
    const float* query,
    const float* data,
    float* results,
    size_t dim,
    size_t num_vectors
) {
    for (size_t v = 0; v < num_vectors; ++v) {
        const float* vec = data + v * dim;

        __m256 sum256 = _mm256_setzero_ps();

        size_t i = 0;
        // Process 16 floats per iteration (2 AVX ops)
        for (; i + 16 <= dim; i += 16) {
            __m256 q1 = _mm256_loadu_ps(query + i);
            __m256 v1 = _mm256_loadu_ps(vec + i);
            sum256 = _mm256_fmadd_ps(q1, v1, sum256);

            __m256 q2 = _mm256_loadu_ps(query + i + 8);
            __m256 v2 = _mm256_loadu_ps(vec + i + 8);
            sum256 = _mm256_fmadd_ps(q2, v2, sum256);
        }

        // Process remaining 8 floats
        for (; i + 8 <= dim; i += 8) {
            __m256 q1 = _mm256_loadu_ps(query + i);
            __m256 v1 = _mm256_loadu_ps(vec + i);
            sum256 = _mm256_fmadd_ps(q1, v1, sum256);
        }

        // Horizontal sum
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0),
                                    _mm256_extractf128_ps(sum256, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float sum = _mm_cvtss_f32(sum128);

        // Handle remainder
        for (; i < dim; ++i) {
            sum += query[i] * vec[i];
        }

        results[v] = sum;
    }
}

// Method 5: Batched 4-way AVX - process 4 vectors simultaneously
void benchmark_batched_avx_4way(
    const float* query,
    const float* data,
    float* results,
    size_t dim,
    size_t num_vectors
) {
    size_t v = 0;
    for (; v + 4 <= num_vectors; v += 4) {
        const float* vec0 = data + (v + 0) * dim;
        const float* vec1 = data + (v + 1) * dim;
        const float* vec2 = data + (v + 2) * dim;
        const float* vec3 = data + (v + 3) * dim;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (size_t i = 0; i + 8 <= dim; i += 8) {
            __m256 q = _mm256_loadu_ps(query + i);

            __m256 d0 = _mm256_loadu_ps(vec0 + i);
            __m256 d1 = _mm256_loadu_ps(vec1 + i);
            __m256 d2 = _mm256_loadu_ps(vec2 + i);
            __m256 d3 = _mm256_loadu_ps(vec3 + i);

            sum0 = _mm256_fmadd_ps(q, d0, sum0);
            sum1 = _mm256_fmadd_ps(q, d1, sum1);
            sum2 = _mm256_fmadd_ps(q, d2, sum2);
            sum3 = _mm256_fmadd_ps(q, d3, sum3);
        }

        // Horizontal sums
        auto hsum = [](__m256 v) -> float {
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(v, 0),
                                        _mm256_extractf128_ps(v, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            return _mm_cvtss_f32(sum128);
        };

        results[v + 0] = hsum(sum0);
        results[v + 1] = hsum(sum1);
        results[v + 2] = hsum(sum2);
        results[v + 3] = hsum(sum3);

        // Handle remainder dimensions
        size_t rem_start = (dim / 8) * 8;
        for (size_t i = rem_start; i < dim; ++i) {
            results[v + 0] += query[i] * vec0[i];
            results[v + 1] += query[i] * vec1[i];
            results[v + 2] += query[i] * vec2[i];
            results[v + 3] += query[i] * vec3[i];
        }
    }

    // Handle remaining vectors
    for (; v < num_vectors; ++v) {
        const float* vec = data + v * dim;
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum += query[i] * vec[i];
        }
        results[v] = sum;
    }
}
#endif

// Method 6: Naive scalar (baseline)
void benchmark_scalar(
    const float* query,
    const float* data,
    float* results,
    size_t dim,
    size_t num_vectors
) {
    for (size_t v = 0; v < num_vectors; ++v) {
        const float* vec = data + v * dim;
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum += query[i] * vec[i];
        }
        results[v] = sum;
    }
}

// Verify results match
bool verify_results(const float* ref_results, const float* test_results, size_t count,
                    const char* name, float tolerance = 1e-4f) {
    float max_diff = 0.0f;
    size_t max_idx = 0;

    for (size_t i = 0; i < count; ++i) {
        float diff = std::abs(ref_results[i] - test_results[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    bool ok = max_diff < tolerance;
    std::cout << "  " << std::left << std::setw(20) << name << ": max_diff=" << std::scientific << max_diff;
    if (!ok) {
        std::cout << " FAIL (ref=" << ref_results[max_idx] << ", test=" << test_results[max_idx] << ")";
    } else {
        std::cout << " OK";
    }
    std::cout << std::fixed << std::endl;

    return ok;
}

//==============================================================================
// Main Benchmark
//==============================================================================

void run_benchmark(size_t dim) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Dimension: " << dim << ", Vectors: " << NUM_VECTORS << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Setup random generator
    std::mt19937 gen(42);

    // Allocate aligned memory
    float* query = aligned_alloc_array<float>(dim);
    float* data = aligned_alloc_array<float>(NUM_VECTORS * dim);
    float* results_simd = aligned_alloc_array<float>(NUM_VECTORS);
    float* results_sgemm = aligned_alloc_array<float>(NUM_VECTORS);
    float* results_sgemv = aligned_alloc_array<float>(NUM_VECTORS);
    float* results_batched = aligned_alloc_array<float>(NUM_VECTORS);
    float* results_batched_4way = aligned_alloc_array<float>(NUM_VECTORS);
    float* results_scalar = aligned_alloc_array<float>(NUM_VECTORS);

    // Generate random data
    generate_random_data(query, dim, gen);
    normalize_vector(query, dim);

    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        generate_random_data(data + i * dim, dim, gen);
        normalize_vector(data + i * dim, dim);
    }

    // Setup hnswlib distance function
    hnswlib::InnerProductSpace space(dim);
    auto dist_func = space.get_dist_func();
    auto dist_param = space.get_dist_func_param();

    // Print SIMD info
    std::cout << "SIMD backend: ";
#if defined(USE_AVX512)
    if (hnswlib::AVX512Capable()) {
        std::cout << "AVX-512";
    } else if (hnswlib::AVXCapable()) {
        std::cout << "AVX2/FMA";
    } else {
        std::cout << "SSE";
    }
#elif defined(USE_AVX)
    if (hnswlib::AVXCapable()) {
        std::cout << "AVX2/FMA";
    } else {
        std::cout << "SSE";
    }
#elif defined(USE_SSE)
    std::cout << "SSE";
#else
    std::cout << "Scalar";
#endif
    std::cout << std::endl;

#ifdef USE_BLAS
    std::cout << "BLAS: Enabled (OpenBLAS)" << std::endl;
#else
    std::cout << "BLAS: Disabled" << std::endl;
#endif

    //--------------------------------------------------------------------------
    // Warmup
    //--------------------------------------------------------------------------
    std::cout << "\nWarming up..." << std::endl;
    for (size_t i = 0; i < NUM_WARMUP; ++i) {
        benchmark_simd_loop(query, data, results_simd, dim, NUM_VECTORS, dist_func, dist_param);
#ifdef USE_BLAS
        benchmark_sgemm(query, data, results_sgemm, dim, NUM_VECTORS);
        benchmark_sgemv(query, data, results_sgemv, dim, NUM_VECTORS);
#endif
#if defined(USE_AVX)
        benchmark_batched_avx(query, data, results_batched, dim, NUM_VECTORS);
        benchmark_batched_avx_4way(query, data, results_batched_4way, dim, NUM_VECTORS);
#endif
        benchmark_scalar(query, data, results_scalar, dim, NUM_VECTORS);
    }

    //--------------------------------------------------------------------------
    // Verify correctness
    //--------------------------------------------------------------------------
    std::cout << "\nVerifying correctness (reference: SIMD loop)..." << std::endl;
    benchmark_simd_loop(query, data, results_simd, dim, NUM_VECTORS, dist_func, dist_param);
#ifdef USE_BLAS
    benchmark_sgemm(query, data, results_sgemm, dim, NUM_VECTORS);
    benchmark_sgemv(query, data, results_sgemv, dim, NUM_VECTORS);
#endif
#if defined(USE_AVX)
    benchmark_batched_avx(query, data, results_batched, dim, NUM_VECTORS);
    benchmark_batched_avx_4way(query, data, results_batched_4way, dim, NUM_VECTORS);
#endif
    benchmark_scalar(query, data, results_scalar, dim, NUM_VECTORS);

#ifdef USE_BLAS
    verify_results(results_simd, results_sgemm, NUM_VECTORS, "SGEMM");
    verify_results(results_simd, results_sgemv, NUM_VECTORS, "SGEMV");
#endif
#if defined(USE_AVX)
    verify_results(results_simd, results_batched, NUM_VECTORS, "Batched AVX");
    verify_results(results_simd, results_batched_4way, NUM_VECTORS, "Batched 4-way");
#endif
    verify_results(results_simd, results_scalar, NUM_VECTORS, "Scalar");

    //--------------------------------------------------------------------------
    // Benchmark
    //--------------------------------------------------------------------------
    std::cout << "\nBenchmarking..." << std::endl;
    std::vector<double> times_simd, times_sgemm, times_sgemv;
    std::vector<double> times_batched, times_batched_4way, times_scalar;

    times_simd.reserve(NUM_ITERATIONS);
    times_sgemm.reserve(NUM_ITERATIONS);
    times_sgemv.reserve(NUM_ITERATIONS);
    times_batched.reserve(NUM_ITERATIONS);
    times_batched_4way.reserve(NUM_ITERATIONS);
    times_scalar.reserve(NUM_ITERATIONS);

    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // SIMD Loop
        {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_simd_loop(query, data, results_simd, dim, NUM_VECTORS, dist_func, dist_param);
            auto end = std::chrono::high_resolution_clock::now();
            times_simd.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

#ifdef USE_BLAS
        // SGEMM
        {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_sgemm(query, data, results_sgemm, dim, NUM_VECTORS);
            auto end = std::chrono::high_resolution_clock::now();
            times_sgemm.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        // SGEMV
        {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_sgemv(query, data, results_sgemv, dim, NUM_VECTORS);
            auto end = std::chrono::high_resolution_clock::now();
            times_sgemv.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
#endif

#if defined(USE_AVX)
        // Batched AVX
        {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_batched_avx(query, data, results_batched, dim, NUM_VECTORS);
            auto end = std::chrono::high_resolution_clock::now();
            times_batched.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        // Batched 4-way
        {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_batched_avx_4way(query, data, results_batched_4way, dim, NUM_VECTORS);
            auto end = std::chrono::high_resolution_clock::now();
            times_batched_4way.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
#endif

        // Scalar
        {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_scalar(query, data, results_scalar, dim, NUM_VECTORS);
            auto end = std::chrono::high_resolution_clock::now();
            times_scalar.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
    }

    //--------------------------------------------------------------------------
    // Calculate and Print Statistics
    //--------------------------------------------------------------------------
    Stats stats_simd = calculate_stats(times_simd);
    Stats stats_scalar = calculate_stats(times_scalar);
#ifdef USE_BLAS
    Stats stats_sgemm = calculate_stats(times_sgemm);
    Stats stats_sgemv = calculate_stats(times_sgemv);
#endif
#if defined(USE_AVX)
    Stats stats_batched = calculate_stats(times_batched);
    Stats stats_batched_4way = calculate_stats(times_batched_4way);
#endif

    // FLOPs calculation: dim * 2 (mul + add) * NUM_VECTORS
    double total_flops = static_cast<double>(dim) * 2.0 * NUM_VECTORS;

    auto print_stats = [&](const char* name, Stats& s) {
        double gflops = total_flops / (s.mean * 1000.0);
        std::cout << std::left << std::setw(22) << name
                  << std::right
                  << " mean: " << std::setw(8) << std::fixed << std::setprecision(2) << s.mean << " µs"
                  << "  median: " << std::setw(8) << s.median << " µs"
                  << "  std: " << std::setw(7) << s.std_dev << " µs"
                  << "  GFLOPS: " << std::setw(6) << gflops
                  << std::endl;
    };

    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Results (lower latency is better):" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    print_stats("SIMD Loop (hnswlib)", stats_simd);
#ifdef USE_BLAS
    print_stats("SGEMM (OpenBLAS)", stats_sgemm);
    print_stats("SGEMV (OpenBLAS)", stats_sgemv);
#endif
#if defined(USE_AVX)
    print_stats("Batched AVX", stats_batched);
    print_stats("Batched 4-way AVX", stats_batched_4way);
#endif
    print_stats("Scalar (baseline)", stats_scalar);

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Speedup vs SIMD Loop (hnswlib):" << std::endl;
#ifdef USE_BLAS
    double sgemm_speedup = stats_simd.mean / stats_sgemm.mean;
    double sgemv_speedup = stats_simd.mean / stats_sgemv.mean;
    std::cout << "  SGEMM:            " << std::setprecision(2) << sgemm_speedup << "x";
    std::cout << (sgemm_speedup > 1 ? " (faster)" : " (slower)") << std::endl;
    std::cout << "  SGEMV:            " << std::setprecision(2) << sgemv_speedup << "x";
    std::cout << (sgemv_speedup > 1 ? " (faster)" : " (slower)") << std::endl;
#endif
#if defined(USE_AVX)
    double batched_speedup = stats_simd.mean / stats_batched.mean;
    double batched_4way_speedup = stats_simd.mean / stats_batched_4way.mean;
    std::cout << "  Batched AVX:      " << std::setprecision(2) << batched_speedup << "x";
    std::cout << (batched_speedup > 1 ? " (faster)" : " (slower)") << std::endl;
    std::cout << "  Batched 4-way:    " << std::setprecision(2) << batched_4way_speedup << "x";
    std::cout << (batched_4way_speedup > 1 ? " (faster)" : " (slower)") << std::endl;
#endif
    std::cout << "  SIMD vs Scalar:   " << std::setprecision(2) << stats_scalar.mean / stats_simd.mean << "x" << std::endl;

    // Cleanup
    std::free(query);
    std::free(data);
    std::free(results_simd);
    std::free(results_sgemm);
    std::free(results_sgemv);
    std::free(results_batched);
    std::free(results_batched_4way);
    std::free(results_scalar);
}

int main(int argc, char* argv[]) {
    std::cout << "===============================================================================" << std::endl;
    std::cout << "       SIMD vs SGEMM Inner Product Micro-Benchmark" << std::endl;
    std::cout << "===============================================================================" << std::endl;
    std::cout << "Scenario: 1 query vector vs " << NUM_VECTORS << " data vectors" << std::endl;
    std::cout << "Warmup: " << NUM_WARMUP << " iterations" << std::endl;
    std::cout << "Benchmark: " << NUM_ITERATIONS << " iterations" << std::endl;
    std::cout << std::endl;
    std::cout << "Methods:" << std::endl;
    std::cout << "  1. SIMD Loop (hnswlib)  - function call per vector" << std::endl;
#ifdef USE_BLAS
    std::cout << "  2. SGEMM (OpenBLAS)     - batch matrix multiply (1,dim) @ (dim,N)" << std::endl;
    std::cout << "  3. SGEMV (OpenBLAS)     - matrix-vector multiply" << std::endl;
#endif
#if defined(USE_AVX)
    std::cout << "  4. Batched AVX          - inline SIMD, no function calls" << std::endl;
    std::cout << "  5. Batched 4-way AVX    - process 4 vectors simultaneously" << std::endl;
#endif
    std::cout << "  6. Scalar               - naive baseline" << std::endl;

    // Test different dimensions
    std::vector<size_t> dimensions = {128, 256, 512, 768, 1024};

    for (size_t dim : dimensions) {
        run_benchmark(dim);
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Benchmark complete!" << std::endl;

    return 0;
}
