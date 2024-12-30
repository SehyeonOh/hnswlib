#include "../../hnswlib/hnswlib.h"
#include "DataToCpp/data2cpp/parquet/parquet2cpp.hh"
#include <atomic>
#include <thread>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] 
                  << " <parquet_path> <column_name> <save_path> <M> <ef_construction> <num_threads>" 
                  << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string parquet_path = argv[1];
    std::string column_name = argv[2];
    std::string save_path = argv[3];
    size_t M = std::stoi(argv[4]);
    size_t ef_construction = std::stoi(argv[5]);
    int num_threads = std::stoi(argv[6]);

    // Use system's thread count if num_threads is 0
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    try {
        // Load Parquet file with specified column name
        data2cpp::Parquet2Cpp data(parquet_path, column_name);
        
        // Extract necessary information from Parquet file
        const size_t dim = data.GetWidth();
        const size_t max_elements = data.GetRowCount();
        
        // Create HNSW index in IP (Inner Product) space
        hnswlib::InnerProductSpace space(dim);
        auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

        // Add vector data to HNSW index using multiple threads
        std::atomic<int64_t> current_index(0);
        std::vector<std::thread> threads;

        for (int thread_id = 0; thread_id < num_threads; thread_id++) {
            threads.emplace_back([&]() {
                while (true) {
                    int64_t i = current_index.fetch_add(1);
                    if (i >= max_elements) break;
                    const float* current_vector = reinterpret_cast<const float*>(data.GetRawData(i));
                    alg_hnsw->addPoint(current_vector, i);
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Save the index
        alg_hnsw->saveIndex(save_path);

        delete alg_hnsw;
        
        std::cout << "Successfully built index with parameters:" << std::endl
                  << "- Dimensions: " << dim << std::endl
                  << "- Max elements: " << max_elements << std::endl
                  << "- M: " << M << std::endl
                  << "- ef_construction: " << ef_construction << std::endl
                  << "- Threads used: " << num_threads << std::endl
                  << "Index saved to: " << save_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
