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


constexpr double agentLocRatio = 2.25;



int main(int argc, char* argv[]) {
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
   
   PostMovement p(NUM_OF_CITIES, NUM_OF_ITERATIONS, agents,  movedRatioInside, movedRatioOutside, locs, print_on);
}





