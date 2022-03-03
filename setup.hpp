#pragma once
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/tuple.h"

#define HD __host__ __device__

class PostMovement {
public:
    static constexpr unsigned AGENT = 0;
    static constexpr unsigned FROM = 1;
    static constexpr unsigned TO = 2;

    thrust::device_vector<unsigned> generatorHelper;
//    thrust::device_vector<unsigned> offset1, offset2; // [0, 0, 2, 3, 3, 3, 4, 4, 4] #localtion_length
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned>> movement1, movement2;
    
    
    //need for setDifference
    thrust::device_vector<unsigned> flags;
    thrust::device_vector<unsigned> scanResult;   
    thrust::device_vector<thrust::pair<unsigned, unsigned>> copyOfPairs;

    std::vector<thrust::device_vector<unsigned>> agentIDs;//contains the IDs in the particular city
    std::vector<thrust::device_vector<thrust::tuple<unsigned, unsigned>>> locationAgentLists;//contains the cityID and the agent ids ordered by locations, not used
    std::vector<thrust::host_vector<unsigned>>hostAgentLocations; // agent ids ordered by locations, this is used
    std::vector<thrust::device_vector<unsigned>> offsets;

    std::vector<thrust::host_vector<unsigned>>placeToCopyAgentLengths;
    std::vector<thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned>>> hostMovements;
    std::vector<thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned>>> hostexChangeAgents;//just for printing
    std::vector<thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned>>> exChangeAgents;
    std::vector<thrust::device_vector<unsigned>> offsetForExChangeAgents;
    std::vector<unsigned> movedAgentSizeFromCities;
    std::vector<thrust::host_vector<unsigned>> exchangeHelperVectors;
    std::vector<thrust::host_vector<unsigned>> stayedAngentsHelperVectors;
    std::vector<thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned >>>IncomingAgents;
    std::vector<thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned >>>hostIncomingAgents;
    std::vector<thrust::device_vector<thrust::tuple<unsigned, unsigned,unsigned>>>agentLocationAfterMovements;




    PostMovement(unsigned NUM_OF_CITIES, unsigned NUM_OF_ITERATIONS,unsigned agents, double movedRatioInside, double movedRatioOutside, unsigned locations,unsigned print_on);

    void print() const;
};