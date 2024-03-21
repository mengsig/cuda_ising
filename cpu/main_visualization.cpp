#include <iostream>
#include <vector>
#include <SDL2/SDL.h>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>


const int GRID_SIZE = 512; //Adjust system size
const int NROWS = GRID_SIZE; //Adjust system size
const int NCOLS = GRID_SIZE; //Adjust system size
const float BETA = 1/1.5; //Adjust this parameter for funny things! Critical Value is BETA = 0.2;
const int CELL_SIZE = 5;
const int SCREEN_WIDTH = GRID_SIZE * CELL_SIZE;
const int SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE;
SDL_Window* gWindow = nullptr;
SDL_Renderer* gRenderer = nullptr;

#define IDX(i, j) (i*NCOLS + j)

//
//
struct Position {
    int row;
    int col;
};

void save_data_to_txt(int* data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            file << data[IDX(i,j)] << " ";
        }
        file << "\n";
    }

    file.close();
}


// Function to initialize SDL
bool init() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    gWindow = SDL_CreateWindow("Abelian Sandpile Model", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (gWindow == nullptr) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    gRenderer = SDL_CreateRenderer(gWindow, -1, SDL_RENDERER_ACCELERATED);
    if (gRenderer == nullptr) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_SetRenderDrawColor(gRenderer, 255, 255, 255, 255);

    return true;
}

// Function to close SDL
void close() {
    SDL_DestroyRenderer(gRenderer);
    SDL_DestroyWindow(gWindow);
    SDL_Quit();
}

// Function to render the grid
const SDL_Color colors[] = {
    {0, 0, 0, 200},     // Black for 0
    {255,255,255, 200},   // white for 1
};

// Function to render the sandpile grid
void renderGrid(int *data) {
    SDL_SetRenderDrawColor(gRenderer, 0, 0, 0, 255);
    SDL_RenderClear(gRenderer);
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j=0; j < GRID_SIZE; j++) {

            int x = j * CELL_SIZE;
            int y = i * CELL_SIZE;

            SDL_Rect cellRect = {x, y, CELL_SIZE, CELL_SIZE};

            // Get the color based on the sandpile value
            SDL_Color color = colors[data[IDX(i,j)]];
            
            SDL_SetRenderDrawColor(gRenderer, color.r, color.g, color.b, color.a);
            SDL_RenderFillRect(gRenderer, &cellRect);

            SDL_SetRenderDrawColor(gRenderer, 0, 0, 0, 255);
            SDL_RenderDrawRect(gRenderer, &cellRect);
        }
    }

    SDL_RenderPresent(gRenderer);
}

int* random_spin_field(int N, int M) {
    int *field = new int[NROWS * NCOLS];;
    //std::vector<std::vector<int>> field(N, std::vector<int>(M));
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            //field[i][j] = (rand() % 2) * 2 - 1; // Randomly choose between -1 and 1
            field[IDX(i,j)] = (rand() % 2) * 2 - 1; // Randomly choose between -1 and 1
        }
    }

    return field;
}

void ising_update(int* field, int n, int m, float beta) {
    int total = 0;
    int N = NROWS;
    int M = NCOLS;
    total += field[IDX((N + n+1)%N, (m+M)%M)];
    total += field[IDX((N + n-1)%N, (m+M)%M)];  
    total += field[IDX((N + n) % N, (M + m+1)%M)];  
    total += field[IDX((N + n) % N, (M + m-1)%M)];  


    
   // for (int i = n - 1; i <= n + 1; ++i) {
   //     for (int j = m - 1; j <= m + 1; ++j) {
   //         if (i == n && j == m) {
   //             continue;
   //         }
   //         //total += field[(i + N) % N][(j + M) % M];
   //         if (i == n || j ==m){
   //           total += field[IDX((i + N) % N, (j + M) % M)];
   //       }
   //     }
   // }

    int dE = 2 * field[IDX(n,m)] * total;
    if (dE <= 0 || (std::exp(-dE * beta) > (float)rand() / RAND_MAX)) {
        field[IDX(n,m)] *= -1;
    }
}

void ising_step(int* field, float beta = 0.4) {
    int N = NROWS;
    int M = NCOLS;
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

void ising_step_random(int* field, float beta = 0.4) {
    int n;
    int m;
    int N = NROWS;
    int M = NCOLS; 
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


    std::srand(std::time(nullptr));
    if (!init()) {
        std::cerr << "Failed to initialize SDL!" << std::endl;
        return 1;
    }
    int* field = random_spin_field(GRID_SIZE, GRID_SIZE);
    float beta = BETA; 
   
    // Game loop
    bool quit = false;
    SDL_Event e;

    while (!quit) {
        ourCount++;
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

    auto start_time1 = std::chrono::high_resolution_clock::now();
    ising_step(field, beta);
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1).count();
    //std::cout << "Update time: " << duration1 << "millseconds" << std::endl;
    
    //here we render
    if (ourCount % 100 == 0) {
      filename = directory_name + "/state_" + std::to_string(ourCount) + ".txt";
      save_data_to_txt(field, filename);
      renderGrid(field);
      std::cout<<ourCount << std::endl;
    }
    SDL_Delay(0.0001); // Adjust the delay for visualization speed
    }

    close();
    return 0;
}
