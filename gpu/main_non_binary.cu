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
const int THREAD_NO = 32;

const int TIME_STEPS = 1000;
const int save_step = TIME_STEPS/10;

const int GRID_SIZE = 2048; // Adjust system size
const int NCOLS = GRID_SIZE;
const int NROWS = GRID_SIZE;

#define IDX(i, j) (i * (NCOLS / 2) + j)

const float BETA = 1/2.25; // Adjust this parameter for funny things! Critical Value is BETA = 0.2;

int calculateChecksum(char* data) {
    int checksum = 0;
    for (int i = 0; i < NROWS; i++)
        for (int j = 0; j < NCOLS; j++) {
            checksum += data[IDX(i,j)];
        }
    return checksum;
}

void save_data_to_txt(char* red, char* black, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < NROWS; ++i) {
        for (int j = 0; j < NCOLS / 2; ++j) {
            file << (int)red[IDX(i,j)] << " ";
            file << (int)black[IDX(i,j)] << " ";
        }
        file << "\n";
    }

    file.close();
}

char* random_spin_field(int N, int M) {
    char* field = new char[N * M];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            field[IDX(i,j)] = (rand() % 2) * 2 - 1; // Randomly choose between -1 and 1
        }
    }

    return field;
}

__global__ void setup_curand_kernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void ising_step(char* red, char* black, float beta, bool is_black, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = idx;
    
    int n = N / (NCOLS / 2);
    int m = N % (NCOLS / 2);
    
    //red + even = -1
    //red + odd = +1
    //black + even = +1
    //black + odd = -1

    //m = m + ((n%2) ^ is_black);

    int col_offset = (n%2)^is_black;
    col_offset = (col_offset << 1) - 1;

    // Check if the cell should be updated in this kernel call
    int total = red[IDX((n + 1) % NROWS, m)] +
                red[IDX((n - 1 + NROWS) % NROWS, m)] +
                red[IDX(n, (m + col_offset + NCOLS/2) % (NCOLS/2))] +
                red[IDX(n, m)];

    float dE = 2 * black[IDX(n,m)] * total;
    float rand_val = curand_uniform(&states[idx]);

    if (dE <= 0 || (expf(-dE * beta) > rand_val)) {
        black[IDX(n,m)] *= -1;
    }
}





int main() {
    std::string directory_name = "GRID_" + std::to_string(GRID_SIZE);
    std::filesystem::create_directory(directory_name);
    srand(SEED);
    char* red_field = random_spin_field(NROWS, NCOLS/2);
    char* black_field = random_spin_field(NROWS, NCOLS/2);
    char* red_deviceField;
    char* black_deviceField;
    cudaMalloc(&red_deviceField, NROWS * NCOLS / 2 * sizeof(char));
    cudaMalloc(&black_deviceField, NROWS * NCOLS / 2 * sizeof(char));
    cudaMemcpy(red_deviceField, red_field, ((NROWS * NCOLS/2)) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(black_deviceField, black_field, ((NROWS * NCOLS/2)) * sizeof(char), cudaMemcpyHostToDevice);

    curandState* states;
    cudaMalloc(&states, NROWS * NCOLS / 2 * sizeof(curandState));
    setup_curand_kernel<<<(NROWS * NCOLS + THREAD_NO - 1) / (2*THREAD_NO), THREAD_NO>>>(states, SEED);
    bool is_black = false;



    auto start_time1 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < TIME_STEPS; k++) {
        
        if (is_black) {
            ising_step<<<(NROWS * NCOLS + THREAD_NO - 1) / (2*THREAD_NO), THREAD_NO>>>(black_deviceField, red_deviceField, BETA, is_black, states);
          is_black = false;
        }else {
            ising_step<<<(NROWS * NCOLS + THREAD_NO - 1) / (2*THREAD_NO), THREAD_NO>>>(red_deviceField, black_deviceField, BETA, is_black, states);
          is_black = true;
        }

//        std::cout << k << "/" << TIME_STEPS << std::endl;

        if (k % save_step == 0) {
            cudaDeviceSynchronize();
            cudaMemcpy(red_field, red_deviceField, NROWS * NCOLS / 2 * sizeof(char), cudaMemcpyDeviceToHost);
            cudaMemcpy(black_field, black_deviceField, NROWS * NCOLS / 2 * sizeof(char), cudaMemcpyDeviceToHost);
            
            std::string filename = directory_name + "/state_" + std::to_string(k) + ".txt";
            save_data_to_txt(red_field, black_field, filename);
        }

    
    }
    cudaDeviceSynchronize();
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1).count();
    std::cout << "Update time: " << duration1 << " milliseconds" << std::endl;
    cudaMemcpy(red_field, red_deviceField, NROWS * NCOLS / 2 * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(black_field, black_deviceField, NROWS * NCOLS / 2 * sizeof(char), cudaMemcpyDeviceToHost);

    std::string filename = directory_name + "/state_" + std::to_string(TIME_STEPS) + ".txt";
    save_data_to_txt(red_field, black_field, filename);

    cudaFree(red_field);
    cudaFree(black_field);
    cudaFree(states);
    cudaDeviceSynchronize();
    std::cout<< "GPU Finished" << std::endl;
    return 0;
}
