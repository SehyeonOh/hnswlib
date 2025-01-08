#include "../../hnswlib/hnswlib.h"
#include "DataToCpp/data2cpp/parquet/parquet2cpp.hh"
#include "DataToCpp/data2cpp/binary/binary2cpp.hh"
#include <atomic>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <fstream>


int main(int argc, char** argv) {
    if (argc != 8) {
        std::cout << "Usage: " << argv[0] 
                  << " <query_parquet> <column_name> <index_path> <gt_save_path> <k> <ef_search> <num_threads>" 
                  << std::endl;
        return 1;
    }

    try {
        // Parse command line arguments
        std::string query_path = argv[1];
        std::string column_name = argv[2];
        std::string index_path = argv[3];
        std::string gt_save_path = argv[4];
        size_t k = std::stoi(argv[5]);
        size_t ef_search = std::stoi(argv[6]);
        int num_threads = std::stoi(argv[7]);

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

        // Load the index
        hnswlib::InnerProductSpace space(dim);
        auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
        alg_hnsw->setEf(ef_search);
        auto row_count = alg_hnsw->cur_element_count.load();

        // Prepare result storage
        std::vector<std::vector<size_t>> results(num_queries, std::vector<size_t>(k));
        
        // Ground truth storage (100 results per query)
        const size_t gt_count = 100;
        std::vector<std::vector<std::pair<float, size_t>>> ground_truth(num_queries, 
            std::vector<std::pair<float, size_t>>(row_count));

        // Multi-threading for ground truth calculation
        std::atomic<size_t> current_query(0);
        std::vector<std::thread> threads;
        
        const size_t chunk_size = 100;  // chunk size for row processing

        // Calculate ground truth
        for (int thread_id = 0; thread_id < num_threads; thread_id++) {
            threads.emplace_back([&]() {
                while (true) {
                    size_t query_idx = current_query.fetch_add(1);
                    if (query_idx >= num_queries) break;

                    const float* query_vector = query_data.GetFloatData(query_idx);
                    
                    // Multi-threading for row processing within each query
                    std::atomic<size_t> current_chunk(0);
                    std::vector<std::thread> row_threads;
                    
                    for (int inner_thread = 0; inner_thread < num_threads; inner_thread++) {
                        row_threads.emplace_back([&]() {
                            while (true) {
                                size_t chunk_idx = current_chunk.fetch_add(1);
                                size_t start_idx = chunk_idx * chunk_size;
                                if (start_idx >= row_count) break;
                                
                                size_t end_idx = std::min(start_idx + chunk_size, row_count);
                                
                                // Calculate distances for each row in the chunk
                                for (size_t j = start_idx; j < end_idx; j++) {
                                    float dist = alg_hnsw->fstdistfunc_(
                                        query_vector, 
                                        alg_hnsw->getDataByInternalId(j), 
                                        alg_hnsw->dist_func_param_
                                    );
                                    ground_truth[query_idx][j] = std::make_pair(dist, alg_hnsw->getExternalLabel(j));
                                }
                            }
                        });
                    }

                    // Wait for row threads to complete
                    for (auto& thread : row_threads) {
                        thread.join();
                    }

                    // Sort by distance and keep top 100
                    std::sort(ground_truth[query_idx].begin(), ground_truth[query_idx].end());
                    ground_truth[query_idx].resize(gt_count);
                }
            });
        }

        // Wait for ground truth calculation to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Save ground truth to binary file
        std::ofstream gt_file(gt_save_path, std::ios::binary);
        if (!gt_file) {
            throw std::runtime_error("Cannot open ground truth file for writing: " + gt_save_path);
        }

        for (size_t i = 0; i < num_queries; i++) {
            for (size_t j = 0; j < gt_count; j++) {
                uint64_t label = static_cast<uint64_t>(ground_truth[i][j].second);
                gt_file.write(reinterpret_cast<const char*>(&label), sizeof(uint64_t));
            }
        }
        gt_file.close();

        std::cout << "Ground truth saved to: " << gt_save_path << std::endl;

        // Variables for HNSW search
        current_query = 0;  // Reset counter
        threads.clear();    // Clear thread vector
        std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> result_queues(num_queries);

        // Perform HNSW search
        for (int thread_id = 0; thread_id < num_threads; thread_id++) {
            threads.emplace_back([&]() {
                while (true) {
                    size_t query_idx = current_query.fetch_add(1);
                    if (query_idx >= num_queries) break;

                    const float* query_vector = query_data.GetFloatData(query_idx);
                    result_queues[query_idx] = alg_hnsw->searchKnn(query_vector, k);
                }
            });
        }

        // Wait for search to complete
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
            for (size_t j = 0; j < k; j++) {
                // Check if results[i][j] exists in first k elements of groundtruth
                for (size_t g = 0; g < k; g++) {
                    if (results[i][j] == ground_truth[i][g].second) {
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
                  << "- Recall@" << k << ": " << recall << std::endl;

        delete alg_hnsw;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
