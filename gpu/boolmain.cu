#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

#include <curand.h>
#include <curand_kernel.h>

const int SEED = 42;
const int THREAD_NO = 64;

const int TIME_STEPS = 1000;
const int save_step = TIME_STEPS/10;

const int GRID_SIZE = 1024; // Adjust system size
const int NCOLS = GRID_SIZE;
const int NROWS = GRID_SIZE;

#define IDX(i, j) (i * NCOLS + j)

const float BETA = 1/2.25; // Adjust this parameter for funny things! Critical Value is BETA = 0.2;

int calculateChecksum(bool* data) {
    int checksum = 0;
    for (int i = 0; i < NROWS; i++)
        for (int j = 0; j < NCOLS; j++) {
            checksum += data[IDX(i,j)];
        }
    return checksum;
}

void save_data_to_txt(bool* data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < NROWS; ++i) {
        for (int j = 0; j < NCOLS; ++j) {
            file << data[IDX(i,j)] << " ";
        }
        file << "\n";
    }

    file.close();
}

bool* random_spin_field(int N, int M) {
    bool* field = new bool[N * M];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            field[IDX(i,j)] = ((rand())%2); // Randomly choose between -1 and 1
        }
    }

    return field;
}

__global__ void setup_curand_kernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void ising_step(bool* field, float beta, bool is_black, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NROWS * NCOLS) {
        int n = idx / NCOLS;
        int m = idx % NCOLS;

        // Determine if the cell is black or white based on its position
        bool is_black_cell = (n + m) % 2 == 0;

        // Check if the cell should be updated in this kernel call
        if ((is_black && is_black_cell) || (!is_black && !is_black_cell)) {
            int total = ((field[IDX((n + 1) % NROWS, m)] << 1) - 1 )+
                        ((field[IDX((n - 1 + NROWS) % NROWS, m)] << 1) - 1 )+
                        ((field[IDX(n, (m + 1) % NCOLS)] << 1) - 1 )+
                        ((field[IDX(n, (m - 1 + NCOLS) % NCOLS)] << 1) - 1 );

            float dE = 2 * ((field[idx] << 1) - 1 ) * total;
            float rand_val = curand_uniform(&states[idx]);

            if (dE <= 0 || (expf(-dE * beta) > rand_val)) {
                //field[idx] *= -1;
                field[idx] = !field[idx];
            }
        }
    }
}





int main() {
    std::string directory_name = "GRID_" + std::to_string(GRID_SIZE);
    std::filesystem::create_directory(directory_name);
    //std::string filename = directory_name + "/evolution_" + std::to_string(index) + ".txt";
    srand(SEED);
    bool* field = random_spin_field(NROWS, NCOLS);
    bool* deviceField;
    cudaMalloc(&deviceField, NROWS * NCOLS * sizeof(bool));
    cudaMemcpy(deviceField, field, NROWS * NCOLS * sizeof(bool), cudaMemcpyHostToDevice);

    curandState* states;
    cudaMalloc(&states, NROWS * NCOLS * sizeof(curandState));
    setup_curand_kernel<<<(NROWS * NCOLS + THREAD_NO - 1) / THREAD_NO, THREAD_NO>>>(states, SEED);
    bool is_black = false;



    auto start_time1 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < TIME_STEPS; k++) {
        ising_step<<<(NROWS * NCOLS + THREAD_NO - 1) / THREAD_NO, THREAD_NO>>>(deviceField, BETA, is_black, states);
        cudaDeviceSynchronize();
        if (is_black) {
          is_black = false;
        }else {
          is_black = true;
        }
        std::cout << k << "/" << TIME_STEPS << std::endl;

        if (k % save_step == 0) {
          cudaMemcpy(field, deviceField, NROWS * NCOLS * sizeof(bool), cudaMemcpyDeviceToHost);
          cudaDeviceSynchronize();
          std::string filename = directory_name + "/state_" + std::to_string(k) + ".txt";
          save_data_to_txt(field, filename);
        }

    
    }
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1).count();
    std::cout << "Update time: " << duration1 << "millseconds" << std::endl;
    cudaMemcpy(field, deviceField, NROWS * NCOLS * sizeof(bool), cudaMemcpyDeviceToHost);

    std::string filename = directory_name + "/state_" + std::to_string(TIME_STEPS) + ".txt";
    save_data_to_txt(field, filename);

    cudaFree(deviceField);
    cudaFree(states);
    cudaDeviceSynchronize();
    std::cout<< "GPU Finished" << std::endl;
    return 0;
}
