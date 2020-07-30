CXX = g++-7
CXXFLAGS = -fopenmp -Wall -Werror -Wextra -pedantic -std=c++17 -g -fsanitize=address
LDFLAGS =  -fopenmp -fsanitize=address

SRC = OpenMP/OpenMP_2dEntropy.cpp utils/samples.cpp main.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = ../bin/main

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(LBLIBS)

clean:
	rm -rf $(OBJ) $(EXEC)