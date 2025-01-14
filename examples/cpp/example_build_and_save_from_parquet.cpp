#include "../../hnswlib/hnswlib.h"
#include "DataToCpp/data2cpp/parquet/parquet2cpp.hh"
#include <nlohmann/json.hpp>
#include <atomic>
#include <thread>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

/*
Example sources.json format:
[
    "/path/to/vectors1.parquet",
    "/path/to/vectors2.parquet",
    "/path/to/vectors3.parquet"
]
*/

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] 
                  << " <sources_json> <column_name> <save_path> <M> <ef_construction> <num_threads>" 
                  << std::endl;
        std::cout << "Note: sources_json should contain an array of parquet file paths" << std::endl;
        std::cout << "Example sources.json: [\"path1.parquet\", \"path2.parquet\"]" << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string json_path = argv[1];
    std::string column_name = argv[2];
    std::string save_path = argv[3];
    size_t M = std::stoi(argv[4]);
    size_t ef_construction = std::stoi(argv[5]);
    int num_threads = std::stoi(argv[6]);

    // Read and parse JSON file
    std::vector<std::string> parquet_paths;
    try {
        std::ifstream f(json_path);
        json j = json::parse(f);
        parquet_paths = j.get<std::vector<std::string>>();
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
        return 1;
    }

    // Use system's thread count if num_threads is 0
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    try {
        // Create Parquet2Cpp instance with multiple sources and single column
        data2cpp::Parquet2Cpp data(parquet_paths, column_name);
        
        // Extract necessary information
        const size_t dim = data.GetWidth();
        const size_t max_elements = data.GetRowCount();
        
        // Create HNSW index
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
                    const float* current_vector = data.GetFloatData(i);
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
