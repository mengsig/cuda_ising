#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

const int TIME_STEPS = 1000;
const int GRID_SIZE = 1024; //Adjust system size
const float BETA = 1/2.23; //Adjust this parameter for funny things! Critical Value is BETA = 0.2;
const int CELL_SIZE = 5;
const int SCREEN_WIDTH = GRID_SIZE * CELL_SIZE;
const int SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE;
//
//
struct Position {
    int row;
    int col;
};


void save_data_to_txt(const std::vector<std::vector<int>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            file << data[i][j] << " ";
        }
        file << "\n";
    }

    file.close();
}

std::vector<std::vector<int>> random_spin_field(int N, int M) {
    std::vector<std::vector<int>> field(N, std::vector<int>(M));
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            field[i][j] = (rand() % 2) * 2 - 1; // Randomly choose between -1 and 1
        }
    }

    return field;
}

void ising_update(std::vector<std::vector<int>>& field, int n, int m, double beta) {
    int total = 0;
    int N = field.size();
    int M = field[0].size();
    
    
    for (int i = n - 1; i <= n + 1; ++i) {
        for (int j = m - 1; j <= m + 1; ++j) {
            if (i == n && j == m) {
                continue;
            }
            total += field[(i + N) % N][(j + M) % M];
        }
    }

    int dE = 2 * field[n][m] * total;
    if (dE <= 0 || (std::exp(-dE * beta) > (double)rand() / RAND_MAX)) {
        field[n][m] *= -1;
    }
}

void ising_step(std::vector<std::vector<int>>& field, float beta = 0.4) {
    int N = field.size();
    int M = field[0].size();
    for (int n_offset = 0; n_offset < 2; ++n_offset) {
        for (int m_offset = 0; m_offset < 2; ++m_offset) {
            for (int n = n_offset; n < GRID_SIZE; n += 2) {
                for (int m = m_offset; m < GRID_SIZE; m += 2) {
                    ising_update(field, n, m, beta);
                }
            }
        }
    }
}

void ising_step_random(std::vector<std::vector<int>>& field, float beta = 0.4) {
    int n;
    int m;
    int N = field.size();
    int M = field[0].size();
  //#pragma omp parallel for 
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++) {
            n =  std::rand() % N;
            m =  std::rand() % M;
            ising_update(field, n, m, beta);
    }
  }
}

int main() {
    int ourCount = 0;
    int index = 0;
    int size = 1;
    std::string directory_name = "GRID_" + std::to_string(GRID_SIZE);
    std::filesystem::create_directory(directory_name);
    std::string filename = directory_name + "/evolution_" + std::to_string(index) + ".txt";
    std::vector<std::vector<int>> field = random_spin_field(GRID_SIZE, GRID_SIZE);
    float beta = BETA; 
   
    // Game loop
    auto start_time1 = std::chrono::high_resolution_clock::now();
    
    for (int k = 0; k < TIME_STEPS; k++) { 
        ising_step(field, beta);
        std::cout << k << "/" << TIME_STEPS << std::endl;
        
    //Uncomment here if you want to save time steps.
    //if (ourCount % 100 == 0) {
    //  filename = directory_name + "/state_" + std::to_string(ourCount) + ".txt";
    //  save_data_to_txt(field, filename);
    //}
    }
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1).count();
    std::cout << "Update time: " << duration1 << "millseconds" << std::endl;
    
    return 0;
}
