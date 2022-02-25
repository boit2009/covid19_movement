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

// std::array<std::pair<unsigned, double>, 8> cases = {{{1000, 0.01}, {1000, 0.1}, {10000, 0.01}, {10000, 0.1},
                                                    // {20000, 0.01}, {20000, 0.1}, {50000, 0.01}, {50000, 0.1}}};


constexpr double agentLocRatio = 2.25;


/*std::tuple<unsigned, unsigned, double, double> getbenchmarkStateInfo(int numOfCities = 1 , int  numOfIterations = 1, int numOfAgents = 100, double movedRatioInside = 0.25, double  MO = 0.05) {
    auto agents = static_cast<unsigned>(NUM_OF_AGENTS);
    auto movedRatioInside = static_cast<double>(movedRatioOutside_);
    auto movedRatioOutside = 1.0 / (10 * static_cast<double>(20));
    auto locs = static_cast<unsigned>(static_cast<double>(agents) / agentLocRatio);
    return std::make_tuple(NUM_OF_CITIES, NUM_OF_ITERATIONS, agents, locs, movedRatioInside, movedRatioOutside);
}

static void benchDirectSort() {
    auto[NUM_OF_CITIES, NUM_OF_ITERATIONS, agents, locs, movedRatioInside, movedRatioOutside] = getbenchmarkStateInfo();
    PostMovement p(NUM_OF_CITIES, NUM_OF_ITERATIONS, agents, locs, movedRatioInside, movedRatioOutside);
  
}*/
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





