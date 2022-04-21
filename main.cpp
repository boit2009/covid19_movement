#include <cstdlib>
#include <sstream>
#include "setup.hpp"
#include <tuple>
#include <array>
#include <iostream>
#include "thrust/transform.h"
#include "thrust/scatter.h"
#include "thrust/set_operations.h"
#include "thrust/iterator/permutation_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <thrust/system/system_error.h>
#include <mpi.h>


constexpr double agentLocRatio = 2.25;



int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if ( size != 1 && size != 2 && size != 4 && size != 8) {
        std::cout << "Must run with 1,2,4 or 8 processes\n";
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    std::stringstream str;
    str << argv[1];
    unsigned NUM_OF_CITIES;
    str >> NUM_OF_CITIES;
    str.clear();
    str << argv[2];
    unsigned NUM_OF_ITERATIONS;
    str >> NUM_OF_ITERATIONS;
    str.clear();
    str << argv[3];
    unsigned agents;
    str >> agents;
    str.clear();
    str << argv[4];
    unsigned print_on;
    str >> print_on;
    auto movedRatioInside=static_cast<double>(0.25);
    auto movedRatioOutside=static_cast<double>(0.05);
    auto locs= static_cast<unsigned>(static_cast<double>(agents) / agentLocRatio);
    std::cout << "used parameters NUM_OF_CITIES: " << NUM_OF_CITIES << " NUM_OF_ITERATIONS: "<<NUM_OF_ITERATIONS << " agents: "<< agents<<
    " movedRatioInside: "<< movedRatioInside << " movedRatioOutside : "<< movedRatioOutside << " locs: "<< locs << " print_on: " << print_on << "\n";
   unsigned rank2 = rank;
   unsigned size2 = size;
   if(NUM_OF_CITIES != size){
       MPI_Abort(MPI_COMM_WORLD,1);
   }

   PostMovement p(NUM_OF_CITIES, NUM_OF_ITERATIONS, agents,  movedRatioInside, movedRatioOutside, locs, print_on, rank2, size2);
}





