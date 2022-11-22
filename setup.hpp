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


    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned >>structForAgents; 
    /* NEW VERSION
    unique agent ID 0
    loc            1
    home partition ID 2
    destination partition ID 3
    iternumber amikor elmegy 4
    iternumber amikor hazajön 5*/
    
    /*OLD VERSION
    unique agent ID 0
    home partition ID 1
    current partition ID 2 
    destination partition ID 3 
    current county ID 4 
    iternumber amikor elmegy 5 
    iternumber amikor hazajön  6*/

    thrust::device_vector<unsigned> generatorHelper;

    thrust::device_vector<unsigned> agentID;//contains the IDs in the particular city
    thrust::device_vector<thrust::tuple<unsigned, unsigned>> locationAgentList;//contains the cityID and the agent ids ordered by locations, not used
    thrust::device_vector<unsigned>hostAgentLocation; // agent ids ordered by locations, this is used
    thrust::device_vector<unsigned> offset;

    thrust::device_vector<unsigned>placeToCopyAgentLength;
    thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> hostexChangeAgent;//just for printing
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> exChangeAgent;
    thrust::device_vector<unsigned> offsetForExChangeAgent;
    unsigned movedAgentSizeFromCity;
    thrust::device_vector<unsigned> exchangeHelperVector;
    thrust::device_vector<unsigned> stayedAngentsHelperVector;
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned >>IncomingAgent;
    thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned >>hostIncomingAgent;
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>>agentLocationAfterMovement;


    /*PostMovement(unsigned NUM_OF_CITIES, int NUM_OF_ITERATIONS,unsigned agents, double movedRatioInside, double movedRatioOutside,
    unsigned locations, unsigned print_on, unsigned rank, unsigned size, unsigned iter_exchange_number);*/
    PostMovement(unsigned NUM_OF_CITIES, int NUM_OF_ITERATIONS, double movedRatioInside, double movedRatioOutside,
    unsigned locations, unsigned print_on, unsigned rank, unsigned size, unsigned iter_exchange_number, unsigned NUM_OF_AGENTS, thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned >> structForAgents,
    thrust::device_vector<unsigned> agentID);


};