CXX = g++
CXXFLAGS = -fopenmp -Wall -Werror -Wextra -pedantic -std=c++17 -g -fsanitize=address
LDFLAGS =  -fopenmp -fsanitize=address

SRC = openmp_HelloWorld.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = ../bin/omp_helloworld

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(LBLIBS)

clean:
	rm -rf $(OBJ) $(EXEC)