CXX = g++
# Memory sanitazion: CXXFLAGS = -fsanitize=address -fno-omit-frame-pointer -g -O2 -fopenmp
CXXFLAGS = -Ofast -fopenmp
# Uncomment for visualiation
#LIBS = -lSDL2main -lSDL2

myprogram: main.cpp
	export OMP_NUM_THREADS=15 && $(CXX) $(CXXFLAGS) -o myprogram main.cpp $(LIBS)

clean:
	rm -f myprogram
