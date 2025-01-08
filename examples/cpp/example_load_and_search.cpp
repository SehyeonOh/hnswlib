#include "../../hnswlib/hnswlib.h"
#include "DataToCpp/data2cpp/parquet/parquet2cpp.hh"
#include "DataToCpp/data2cpp/binary/binary2cpp.hh"
#include <atomic>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cout << "Usage: " << argv[0] 
                  << " <query_parquet> <column_name> <groundtruth_bin> <gt_count_per_query> <index_path> <k> <ef_search> <num_threads>" 
                  << std::endl;
        return 1;
    }

    try {
        // Parse command line arguments
        std::string query_path = argv[1];
        std::string column_name = argv[2];
        std::string gt_path = argv[3];
        size_t gt_count_per_query = std::stoi(argv[4]);
        std::string index_path = argv[5];
        size_t k = std::stoi(argv[6]);
        size_t ef_search = std::stoi(argv[7]);
        int num_threads = std::stoi(argv[8]);

        // Use system's thread count if num_threads is 0
        if (num_threads <= 0) {
            num_threads = std::thread::hardware_concurrency();
        }

        // Load query vectors from parquet
        std::vector<std::string> query_paths;
        query_paths.push_back(query_path);
        data2cpp::Parquet2Cpp query_data(query_paths, column_name);
        const size_t dim = query_data.GetWidth();
        const size_t num_queries = query_data.GetRowCount();

        // Load groundtruth from binary using parameters from query data
        data2cpp::Binary2Cpp gt_data(gt_path, gt_count_per_query, sizeof(uint64_t));
        
        // Load the index
        hnswlib::InnerProductSpace space(dim);
        auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
        alg_hnsw->setEf(ef_search);

        // Prepare result storage
        std::vector<std::vector<size_t>> results(num_queries, std::vector<size_t>(k));
        
        // Distribute search tasks among threads
        std::atomic<size_t> current_query(0);
        std::vector<std::thread> threads;
        std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> result_queues(num_queries);

        for (int thread_id = 0; thread_id < num_threads; thread_id++) {
            threads.emplace_back([&]() {
                while (true) {
                    size_t query_idx = current_query.fetch_add(1);
                    if (query_idx >= num_queries) break;

                    // Search for current query
                    const float* query_vector = query_data.GetFloatData(query_idx);
                    result_queues[query_idx] = alg_hnsw->searchKnn(query_vector, k);
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Convert all priority queues to results vector
        for (size_t i = 0; i < num_queries; i++) {
            auto& queue = result_queues[i];
            for (int j = k - 1; j >= 0; j--) {
                results[i][j] = queue.top().second;
                queue.pop();
            }
        }

        // Compare with groundtruth and calculate recall
        size_t correct_count = 0;
        for (size_t i = 0; i < num_queries; i++) {
            const uint64_t* gt = reinterpret_cast<const uint64_t*>(gt_data.GetRawData(i));
            for (size_t j = 0; j < k; j++) {
                // Check if results[i][j] exists in first k elements of groundtruth
                for (size_t g = 0; g < k; g++) {
                    if (results[i][j] == gt[g]) {
                        correct_count++;
                        break;  // Found a match, move to next result
                    }
                }
            }
        }

        float recall = static_cast<float>(correct_count) / (num_queries * k);

        std::cout << "Search completed with parameters:" << std::endl
                  << "Index parameters:" << std::endl
                  << "- M: " << alg_hnsw->M_ << std::endl
                  << "- ef_construction: " << alg_hnsw->ef_construction_ << std::endl
                  << "- Current element count: " << alg_hnsw->cur_element_count << std::endl
                  << "- Maximum element count: " << alg_hnsw->max_elements_ << std::endl
                  << std::endl
                  << "Search parameters:" << std::endl
                  << "- Number of queries: " << num_queries << std::endl
                  << "- ef_search: " << ef_search << std::endl
                  << "- k: " << k << std::endl
                  << "- Threads used: " << num_threads << std::endl
                  << "- Recall@" << k << ": " << recall << std::endl
                  << "- Distance calculations: " << hnswlib::dist_calculations << std::endl
                  << "- Distance calculations per query: " << hnswlib::dist_calculations / num_queries << std::endl
                  << "- Hops: " << hnswlib::hops << std::endl
                  << "- Hops per query: " << hnswlib::hops / num_queries << std::endl;

        delete alg_hnsw;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
