#include "setup.hpp"
#include <random>
#include <chrono>
#include <ctime>
#include <thrust/for_each.h>
#include "thrust/host_vector.h"
#include "thrust/generate.h"
#include "thrust/copy.h"
#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/transform.h"
#include "thrust/scan.h"
#include "thrust/count.h"
#include <iostream>
#include "thrust/iterator/zip_iterator.h"
#include "thrust/iterator/permutation_iterator.h"
#include <thrust/binary_search.h>
#include <mpi.h>
#include "randomGenerator.h"
#ifdef NVTX
#include "nvToolsExt.h"
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif


std::vector<std::mt19937_64> RandomGenerator::generators;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda.h>
#include <curand_kernel.h>
//__device__ curandState* dstates;
unsigned dstates_size = 0;
__global__ void setup_kernel(unsigned total, curandState* dstates2) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    if (id < total) curand_init(1234, id, 0, &dstates2[id]);
}
#endif

void RandomGenerator::init(unsigned agents) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    curandState* devStates;
    cudaMalloc((void**)&devStates, agents * sizeof(curandState));
    cudaMemcpyToSymbol(dstates, &devStates, sizeof(curandState*));
    dstates_size = agents;
    setup_kernel<<<(agents - 1) / 128 + 1, 128>>>(agents, devStates);
    cudaDeviceSynchronize();
#endif
    unsigned threads = omp_get_max_threads();
    generators.reserve(threads);
    std::random_device rd;
    for (unsigned i = 0; i < threads; ++i) { generators.emplace_back(rd()); }
}

void RandomGenerator::resize(unsigned agents) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    if (dstates_size < agents) {
        curandState* devStates;
        cudaMalloc((void**)&devStates, agents * sizeof(curandState));
        curandState* devStates_old;
        cudaMemcpyFromSymbol(&devStates_old, dstates, sizeof(curandState*));
        cudaMemcpy(devStates, devStates_old, dstates_size * sizeof(curandState), cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(dstates, &devStates, sizeof(curandState*));
        setup_kernel<<<(agents - dstates_size - 1) / 128 + 1, 128>>>(agents - dstates_size, devStates + dstates_size);
        dstates_size = agents;
        cudaDeviceSynchronize();
    }
#endif
    if (generators.size() < omp_get_max_threads()) {
        unsigned threads = omp_get_max_threads();
        generators.reserve(threads);
        std::random_device rd;
        for (unsigned i = generators.size(); i < threads; ++i) { generators.emplace_back(rd()); }
    }
}


struct ZipComparatorToDest
{
    __host__ __device__
    inline bool operator() (const thrust::tuple<unsigned , unsigned, unsigned, unsigned , unsigned, unsigned> &a, const thrust::tuple<unsigned , unsigned, unsigned, unsigned , unsigned, unsigned> &b)
    {
        return thrust::get<3>(a) < thrust::get<3>(b);
    }
};

struct ZipComparatorToHome
{
    __host__ __device__
    inline bool operator() (const thrust::tuple<unsigned , unsigned, unsigned, unsigned , unsigned, unsigned> &a, const thrust::tuple<unsigned , unsigned, unsigned, unsigned , unsigned, unsigned> &b)
    {
        return thrust::get<2>(a) < thrust::get<2>(b);
    }
};
void helperFunction(unsigned NUM_OF_CITIES, int NUM_OF_ITERATIONS, double movedRatioInside, double movedRatioOutside, unsigned locations, unsigned print_on,
    thrust::device_vector<unsigned> &generatorHelper,
    thrust::device_vector<unsigned> &agentID,
    thrust::device_vector<thrust::tuple<unsigned, unsigned>> &locationAgentList,
    thrust::device_vector<unsigned>&hostAgentLocation,
    thrust::device_vector<unsigned> &offset,
    thrust::device_vector<unsigned>&placeToCopyAgentLength,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> &hostMovement,
    thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> &hostexChangeAgent,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> &exChangeAgent,
    thrust::device_vector<unsigned> &offsetForExChangeAgent,
    unsigned &movedAgentSizeFromCity,
    thrust::device_vector<unsigned> &exchangeHelperVector,
    thrust::device_vector<unsigned> &stayedAngentsHelperVector,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>>&IncomingAgent,
    thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>>&hostIncomingAgent,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>>&agentLocationAfterMovement,
    unsigned rank,
    unsigned size,
    unsigned iter_exchange_number,
    unsigned NUM_OF_AGENTS,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned >> &structForAgents
    ){
    
    MPI_Request *requests =new MPI_Request[size*2 - 2];


    RandomGenerator::init(NUM_OF_AGENTS);
    unsigned locationNumberPerCity = (locations / size) - 1; 
    if(print_on)
        printf("The_locationNumberPerCity_value_is:, %d \n", locationNumberPerCity);
    auto t01 = std::chrono::high_resolution_clock::now();
    //reserving memory 
#ifdef NVTX
    nvtxRangePushA("init_host_data");
#endif
    locationAgentList.reserve(NUM_OF_AGENTS*2.0);
    hostAgentLocation.reserve(NUM_OF_AGENTS*2.0);
    offset.resize((locations/size) + 1);
    placeToCopyAgentLength.reserve(NUM_OF_AGENTS*2.0);
    hostMovement.reserve(NUM_OF_AGENTS*2.0);
    hostexChangeAgent.reserve(NUM_OF_AGENTS*2.0);
    exChangeAgent.reserve(NUM_OF_AGENTS*2.0);
    offsetForExChangeAgent.reserve(size+1);
    offsetForExChangeAgent.resize(size+1);
    exchangeHelperVector.reserve(NUM_OF_AGENTS*2.0);
    stayedAngentsHelperVector.reserve(NUM_OF_AGENTS*2.0);
    IncomingAgent.reserve(NUM_OF_AGENTS*2.0);
    hostIncomingAgent.reserve(NUM_OF_AGENTS*2.0);
    agentLocationAfterMovement.reserve(NUM_OF_AGENTS*2.0);
#ifdef NVTX
    nvtxRangePop();
#endif
    
    generatorHelper.reserve(NUM_OF_AGENTS/size*2.0);
    unsigned vector_size;
    
    
    
    //firstly generate the locations and the agentIDs
    vector_size= NUM_OF_AGENTS;
    //agentID.resize(vector_size); done in main
/*#ifdef NVTX
    nvtxRangePushA("genereate_sequences");
#endif
    thrust::sequence(agentID.begin(), agentID.end(), 0 + rank *  vector_size);
#ifdef NVTX
    nvtxRangePop();
#endif*/
    hostAgentLocation.resize(vector_size);
#ifdef NVTX
    nvtxRangePushA("genereate_random locations");
#endif
    thrust::generate(hostAgentLocation.begin(), hostAgentLocation.end(), [locationNumberPerCity] __host__ __device__(){
        return RandomGenerator::randomUnsigned(locationNumberPerCity);
    });
#ifdef NVTX
    nvtxRangePop();
#endif
    if(print_on){
        std::cout << "\n Generated locations\n ";
        thrust::copy(hostAgentLocation.begin(), hostAgentLocation.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        std::cout << "\n ";
    }



    
    auto t02 = std::chrono::high_resolution_clock::now();
    
    
    auto update_arrays_time = 0;
    auto movement_time = 0;
    auto picking_out_stayed_exchanged_agents = 0;
    auto exchanging_agents_with_mpi = 0;
    auto create_the_new_arrays_after_movement = 0;
    auto copying_agents_when_no_communication = 0;
    auto sorting_and_picking_exchanging_agents = 0;
    auto copying_agents_to_agentLocationAfterMovement = 0;


    auto sum1 = std::chrono::high_resolution_clock::now();
    for(int ITER=-1;ITER<NUM_OF_ITERATIONS;ITER++){
        if(rank==0)
            std::cout<<" ITER number "<< ITER<<std::endl; 
        if(ITER == 0){
            update_arrays_time = 0;
            movement_time = 0;
            picking_out_stayed_exchanged_agents = 0;
            exchanging_agents_with_mpi = 0;
            create_the_new_arrays_after_movement = 0;
            copying_agents_when_no_communication = 0;
            sorting_and_picking_exchanging_agents = 0;
            copying_agents_to_agentLocationAfterMovement = 0;
            sum1 = std::chrono::high_resolution_clock::now();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        placeToCopyAgentLength.resize(vector_size);
        locationAgentList.resize(vector_size);
        generatorHelper.resize(vector_size);
        hostAgentLocation.resize(vector_size); 

#ifdef NVTX
        nvtxRangePushA("copy_the_host_agent_locations_to_copy_array_in_the_beginning");             
#endif
        thrust::copy(hostAgentLocation.begin(), hostAgentLocation.end(), placeToCopyAgentLength.begin());
#ifdef NVTX
        nvtxRangePop();
        nvtxRangePushA("sequence_generation_before_sorting_in_the_beginning"); 
#endif
        thrust::sequence(generatorHelper.begin(), generatorHelper.end());
#ifdef NVTX
	nvtxRangePop();
    	nvtxRangePushA("sorting_arrays_in_the_beginning"); 
#endif
        thrust::stable_sort_by_key(placeToCopyAgentLength.begin(), placeToCopyAgentLength.end(), generatorHelper.begin());
#ifdef NVTX
        nvtxRangePop();
#endif
        if(print_on){
            /*std::cout << "Generate locationagentlist \n ";
            thrust::copy(generatorHelper.begin(), generatorHelper.end(), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << "\n";
            thrust::copy(placeToCopyAgentLength.begin(), placeToCopyAgentLength.end(), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << "\n";*/
        }
#ifdef NVTX
        nvtxRangePushA("sequence_generation_before_lower_bound_in_the_beginning"); 
#endif
        thrust::sequence(offset.begin(), offset.end());
#ifdef NVTX
        nvtxRangePop();
        nvtxRangePushA("lower_bound_time_in_the_beginning"); 
#endif
        thrust::lower_bound(placeToCopyAgentLength.begin(),placeToCopyAgentLength.end(),offset.begin(), offset.end(),offset.begin());
#ifdef NVTX
        nvtxRangePop();
#endif


        
        if (print_on){
            /*std::cout<<"\n";
            thrust::copy(offset.begin(), offset.end(), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout<<"\n";*/
        }

        
        auto t2 = std::chrono::high_resolution_clock::now();
        update_arrays_time += std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();


        
        // generate movement
        auto t3 = std::chrono::high_resolution_clock::now();
        agentID.resize(vector_size);
        exchangeHelperVector.resize(vector_size);
        stayedAngentsHelperVector.resize(vector_size);
#ifdef NVTX
        nvtxRangePushA("fill_helper_vectos_with_zeros"); 
#endif
        thrust::fill(exchangeHelperVector.begin(), exchangeHelperVector.end(), 0);
        thrust::fill(stayedAngentsHelperVector.begin(), stayedAngentsHelperVector.end(), 0);                       
#ifdef NVTX
       nvtxRangePop();
#endif
        structForAgents.resize(vector_size);
#ifdef NVTX
        nvtxRangePushA("generate_the_random_movement");
#endif
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(agentID.begin(), hostAgentLocation.begin(),structForAgents.begin()))
                        , thrust::make_zip_iterator(thrust::make_tuple(agentID.end(), hostAgentLocation.end(),structForAgents.end()))
                        , structForAgents.begin()
                        , [rank, movedRatioOutside, movedRatioInside, NUM_OF_CITIES, locationNumberPerCity, print_on, ITER, iter_exchange_number]  __host__ __device__ (thrust::tuple<unsigned&, unsigned&,
                        thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>&> id_loc_struct_agent_trio)  {
            unsigned idx = thrust::get<0>(id_loc_struct_agent_trio);
            unsigned loc = thrust::get<1>(id_loc_struct_agent_trio);
            thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned> structForAgent = thrust::get<2>(id_loc_struct_agent_trio);
            unsigned iter_number_agent_go_to_destination = thrust::get<4>(structForAgent);
            unsigned iter_number_agent_go_home = thrust::get<5>(structForAgent);
            bool agent_going_destination_partition = (iter_number_agent_go_to_destination == (unsigned(ITER)));
            bool agent_going_home_partition = (iter_number_agent_go_home == (unsigned(ITER)));
            int difference_between_iterations_and_dest_time_dest = ITER - iter_number_agent_go_to_destination;
            int difference_between_iterations_and_dest_time_home = ITER - iter_number_agent_go_home;
            bool agent_will_go_dest_in_next_MPI_comm = difference_between_iterations_and_dest_time_dest > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_dest;
            bool agent_will_go_home_in_next_MPI_comm = difference_between_iterations_and_dest_time_home > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_home;

            if(agent_will_go_dest_in_next_MPI_comm  || agent_will_go_home_in_next_MPI_comm )// agent will go dest in the next MPIcomm
            {
                if( agent_going_destination_partition || agent_going_home_partition ) { //if agent go to other city in current step
                    unsigned newLoc =  locationNumberPerCity-1; 
                    return thrust::make_tuple(idx ,newLoc, thrust::get<2>(structForAgent),thrust::get<3>(structForAgent),thrust::get<4>(structForAgent),thrust::get<5>(structForAgent));
                    }
                else
                    return structForAgent;

            }

            double generatedRandom = RandomGenerator::randomUnit();   
            if(generatedRandom < movedRatioInside) { //if agent goes inside the partition     
                unsigned newLoc =  RandomGenerator::randomUnsigned(locationNumberPerCity);
                while(newLoc == loc) { newLoc =  RandomGenerator::randomUnsigned(locationNumberPerCity); }
                    return thrust::make_tuple(idx ,newLoc, thrust::get<2>(structForAgent),thrust::get<3>(structForAgent),thrust::get<4>(structForAgent),thrust::get<5>(structForAgent));
                }   
                    return structForAgent; // it will never happen
        });
        auto t4 = std::chrono::high_resolution_clock::now();
        movement_time += std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();
        if ((ITER+1)%iter_exchange_number==0){
#ifdef NVTX
            nvtxRangePop();
#endif
            
            auto t4_1 = std::chrono::high_resolution_clock::now(); 
                
            
            //  std::cout << "\n";   
            movedAgentSizeFromCity=0;
#ifdef NVTX
         nvtxRangePushA("fill_the_helper_vectors_based_on_the_movement");
#endif
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                                    structForAgents.begin(),
                                    exchangeHelperVector.begin(),
                                    stayedAngentsHelperVector.begin())),
                                    thrust::make_zip_iterator(thrust::make_tuple(
                                    structForAgents.end(),
                                    exchangeHelperVector.end(),
                                    stayedAngentsHelperVector.end())),
                                    [rank, ITER, iter_exchange_number] __host__ __device__ ( thrust::tuple<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>&, unsigned&, unsigned&> tup) {
                thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned> structForAgent = thrust::get<0>(tup);                        
                unsigned iter_number_agent_go_to_destination = thrust::get<4>(structForAgent);
                unsigned iter_number_agent_go_home = thrust::get<5>(structForAgent);
                bool agent_going_destination_partition = (iter_number_agent_go_to_destination == (unsigned(ITER)));
                bool agent_going_home_partition = (iter_number_agent_go_home == (unsigned(ITER)));
                int difference_between_iterations_and_dest_time_dest = ITER - iter_number_agent_go_to_destination;
                int difference_between_iterations_and_dest_time_home = ITER - iter_number_agent_go_home;
                bool agent_will_go_dest_in_next_MPI_comm = difference_between_iterations_and_dest_time_dest > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_dest;
                bool agent_will_go_home_in_next_MPI_comm = difference_between_iterations_and_dest_time_home > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_home;
                if (agent_will_go_dest_in_next_MPI_comm || agent_will_go_home_in_next_MPI_comm){//find out which agent will leave the city
                    thrust::get<1>(tup)=1;
                }else{//the others will stay
                    thrust::get<2>(tup)=1;
                }
            }); 
#ifdef NVTX
          nvtxRangePop();
#endif

        
            auto t4_2 = std::chrono::high_resolution_clock::now();
            picking_out_stayed_exchanged_agents += std::chrono::duration_cast<std::chrono::microseconds>(t4_2-t4_1).count(); 
            auto t5 = std::chrono::high_resolution_clock::now(); 
            
            
            unsigned exchangedAgents = exchangeHelperVector[structForAgents.size()-1];
#ifdef NVTX
         nvtxRangePushA("exclusive_scan_with_exchange_helper_vectors");
#endif
            thrust::exclusive_scan(exchangeHelperVector.begin(), exchangeHelperVector.end(), exchangeHelperVector.begin());
#ifdef NVTX
          nvtxRangePop();
#endif
            movedAgentSizeFromCity = exchangedAgents + exchangeHelperVector[structForAgents.size()-1]; //to know how many agent left city
            exChangeAgent.resize(movedAgentSizeFromCity);
            if(print_on)
                hostexChangeAgent.resize(movedAgentSizeFromCity);
#ifdef NVTX
        nvtxRangePushA("put_leaving_agents_into_exchangeagents");
#endif
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(// this for each will put the leaving agents into the exChangeAgents[i].
                        structForAgents.begin(),
                        thrust::make_permutation_iterator(exChangeAgent.begin(), exchangeHelperVector.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                        structForAgents.end(),
                        thrust::make_permutation_iterator(exChangeAgent.begin(), exchangeHelperVector.end()))),
                [rank,  ITER, iter_exchange_number] __host__ __device__ ( thrust::tuple<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>&, thrust::tuple<unsigned, unsigned,unsigned, unsigned, unsigned, unsigned>&> tup) {
                thrust::tuple<unsigned, unsigned,unsigned, unsigned, unsigned, unsigned> structForAgent =thrust::get<0>(tup);
                unsigned iter_number_agent_go_to_destination = thrust::get<4>(structForAgent);
                unsigned iter_number_agent_go_home = thrust::get<5>(structForAgent);
                bool agent_going_destination_partition = (iter_number_agent_go_to_destination == (unsigned(ITER)));
                bool agent_going_home_partition = (iter_number_agent_go_home == (unsigned(ITER)));
                int difference_between_iterations_and_dest_time_dest = ITER - iter_number_agent_go_to_destination;
                int difference_between_iterations_and_dest_time_home = ITER - iter_number_agent_go_home;
                bool agent_will_go_dest_in_next_MPI_comm = difference_between_iterations_and_dest_time_dest > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_dest;
                bool agent_will_go_home_in_next_MPI_comm = difference_between_iterations_and_dest_time_home > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_home;
                if (agent_will_go_dest_in_next_MPI_comm || agent_will_go_home_in_next_MPI_comm){//find out which agent will leave the city and put to exchanging agents
                        thrust::get<1>(tup) = structForAgent;
                        /*thrust::make_tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>(thrust::get<0>(structForAgent),
                        thrust::get<1>(structForAgent),thrust::get<2>(structForAgent),thrust::get<3>(structForAgent),
                        thrust::get<4>(structForAgent),thrust::get<5>(structForAgent));*/ //put the values to exchangeagents
                    }
            });
#ifdef NVTX
            nvtxRangePop();
#endif
            if(print_on)
                hostexChangeAgent = exChangeAgent; // to be able to print, this can be removed later
#ifdef NVTX
            nvtxRangePushA("stable_sort_on_exchange_agnets");
#endif
            if(ITER < NUM_OF_ITERATIONS/2)
                thrust::stable_sort(exChangeAgent.begin(),exChangeAgent.end(), ZipComparatorToDest()); //sort the exchanging array by dest city
            else
                thrust::stable_sort(exChangeAgent.begin(),exChangeAgent.end(), ZipComparatorToHome()); //sort the exchanging array by home city
#ifdef NVTX
         nvtxRangePop();
#endif
            thrust::device_vector<unsigned>cityIndex(exChangeAgent.size());
#ifdef NVTX
          nvtxRangePushA("pull_out_the_needed_city_indexes");
#endif
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                        cityIndex.begin(),
                        exChangeAgent.begin()))
                        ,thrust::make_zip_iterator(thrust::make_tuple(
                        cityIndex.end(),
                        exChangeAgent.end())),[ITER, NUM_OF_ITERATIONS] __host__ __device__ (thrust::tuple<unsigned&, thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>&> tup) {
                            thrust::tuple<unsigned, unsigned, unsigned,unsigned, unsigned, unsigned> &structForAgent =thrust::get<1>(tup);
                            if(ITER < NUM_OF_ITERATIONS/2)
                                thrust::get<0>(tup) = thrust::get<3>(structForAgent);//put the destination city indexes into cityIndex array
                            else    
                                thrust::get<0>(tup) = thrust::get<2>(structForAgent);//put the home city indexes into cityIndex array
                            
                });
                
#ifdef NVTX
         nvtxRangePop();
         nvtxRangePushA("sequence_before_offsetForExChangeAgents_lower_bound");
#endif
            thrust::sequence(offsetForExChangeAgent.begin(), offsetForExChangeAgent.end());
#ifdef NVTX
          nvtxRangePop();
          nvtxRangePushA("lower_bound_on_offsetForExChangeAgents");
#endif
            thrust::lower_bound(cityIndex.begin(),cityIndex.end(),offsetForExChangeAgent.begin(),
            offsetForExChangeAgent.end(),offsetForExChangeAgent.begin());
#ifdef NVTX
            nvtxRangePop();
#endif

        
#ifdef NVTX
	    nvtxRangePushA("move exchange data to host");
#endif 
            hostexChangeAgent.resize(exChangeAgent.size());    
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
            thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>* gpuptr = thrust::raw_pointer_cast(&exChangeAgent[0]);
            thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>* cpuptr = thrust::raw_pointer_cast(&hostexChangeAgent[0]);
            cudaMemcpy(cpuptr,gpuptr,exChangeAgent.size()*sizeof( thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>),cudaMemcpyDeviceToHost); 
#else       
            thrust::copy(exChangeAgent.begin(), exChangeAgent.end(), hostexChangeAgent.begin());
#endif

#ifdef NVTX
	    nvtxRangePop();
#endif
            auto t6 = std::chrono::high_resolution_clock::now(); 
            sorting_and_picking_exchanging_agents += std::chrono::duration_cast<std::chrono::microseconds>(t6-t5).count(); 
            auto t7 = std::chrono::high_resolution_clock::now(); 
            unsigned counter=0;
            unsigned incomingAgentsNumberToParticularCity=0;
            unsigned *incoming_agents_array = new unsigned[size];
#ifdef NVTX
	    nvtxRangePushA("send/recv counts");
#endif  
        if (ITER != -1){ //there is a needed communication for proper toime measurement
            for(unsigned j = 0; j < size;j++){
                if(j !=rank){
                    unsigned number_of_sent_agents = offsetForExChangeAgent[j+1] - offsetForExChangeAgent[j]; 
                    MPI_Isend(&number_of_sent_agents, 1, MPI_UNSIGNED, j, rank, MPI_COMM_WORLD, &requests[2*counter + 0]);
                    MPI_Irecv(&incoming_agents_array[j], 1, MPI_UNSIGNED, j, j, MPI_COMM_WORLD, &requests[2*counter + 1]);
                    counter++;
                }
            }
            MPI_Waitall(2*size - 2, &requests[0], MPI_STATUSES_IGNORE);
            for(unsigned j = 0; j < size;j++){
                if(j !=rank){
                    incomingAgentsNumberToParticularCity +=incoming_agents_array[j];
                }
            }
        }else{// thiis is only needed for proper time emasurement, this will only send 1-s to all the other processes
             for(unsigned j = 0; j < size;j++){
                if(j !=rank){
                    unsigned number_of_sent_agents = 1;
                    MPI_Isend(&number_of_sent_agents, 1, MPI_UNSIGNED, j, rank, MPI_COMM_WORLD, &requests[2*counter + 0]);
                    MPI_Irecv(&incoming_agents_array[j], 1, MPI_UNSIGNED, j, j, MPI_COMM_WORLD, &requests[2*counter + 1]);
                    counter++;
                }
            }
            MPI_Waitall(2*size - 2, &requests[0], MPI_STATUSES_IGNORE);

        }
            
            
#ifdef NVTX
	    nvtxRangePop();
	    nvtxRangePushA("send/recv agents");
#endif
            
            hostIncomingAgent.resize(incomingAgentsNumberToParticularCity);
            counter=0;
            if (ITER != -1){
                for(unsigned j = 0; j < size;j++){
                    if(j !=rank){
                        unsigned number_of_sent_agents = offsetForExChangeAgent[j+1] - offsetForExChangeAgent[j];
                        MPI_Isend((unsigned*)&hostexChangeAgent[offsetForExChangeAgent[j]], 6*number_of_sent_agents, MPI_UNSIGNED, j, rank, MPI_COMM_WORLD, &requests[2*counter + 0]);
                        unsigned where_to_start=0;
                        for(int i=0;i<j;i++){
                            if(i!=rank){
                                where_to_start+= incoming_agents_array[i];
                            }
                        }
                        MPI_Irecv(&hostIncomingAgent[where_to_start],6*incoming_agents_array[j], MPI_UNSIGNED, j, j, MPI_COMM_WORLD, &requests[2*counter + 1]);
                        counter++;
                    }
                }
                MPI_Waitall(2*size - 2, &requests[0], MPI_STATUSES_IGNORE); 
                delete[] incoming_agents_array;
            }
            auto t8 = std::chrono::high_resolution_clock::now(); 
            exchanging_agents_with_mpi += std::chrono::duration_cast<std::chrono::microseconds>(t8-t7).count();
            auto t9 = std::chrono::high_resolution_clock::now(); 
#ifdef NVTX
	    nvtxRangePop();
#endif        
        
            if(print_on){
                    hostexChangeAgent = exChangeAgent; // to be able to print, this can be removed later
                std::cout <<"After sorting: Agents moving from "<< rank << "st city :"<< movedAgentSizeFromCity <<"\n ";
                for(unsigned  j =0;j<  hostexChangeAgent.size();j++){
                    unsigned id =thrust::get<0>(hostexChangeAgent[j]);
                    unsigned loc =thrust::get<1>(hostexChangeAgent[j]);
                    unsigned home =thrust::get<2>(hostexChangeAgent[j]);
                    unsigned dest =thrust::get<3>(hostexChangeAgent[j]);
                    unsigned go =thrust::get<4>(hostexChangeAgent[j]);
                    unsigned arrieve =thrust::get<5>(hostexChangeAgent[j]);
                    std::cout<<  " id" <<id << " loc " << loc<< " home " << home <<  " dest" <<dest << " go " << go<< " arrieve  " << arrieve<< "\n"; ;
                }
                std::cout << "\n";
            }
            // in the end we have the exchanging arrays and the exchanging offset array
    
            
            if(print_on)
                std::cout<<"\n";

        
            //generate the locations after the movements
            if(print_on)
                std::cout<<"incomingAgentsNumberToParticularCity "<<incomingAgentsNumberToParticularCity<<"\n";

#ifdef NVTX
          nvtxRangePushA("resize incoming GPU buffers");
#endif
            IncomingAgent.resize(incomingAgentsNumberToParticularCity);
            //the new size will be the original size - left agents size + incoming agents size
            agentLocationAfterMovement.resize(vector_size-movedAgentSizeFromCity+incomingAgentsNumberToParticularCity);
#ifdef NVTX
	  nvtxRangePop();
          nvtxRangePushA("exclusive_scan_on_stayed_helper_vector");
#endif
            thrust::exclusive_scan(stayedAngentsHelperVector.begin(), stayedAngentsHelperVector.end(), stayedAngentsHelperVector.begin());
#ifdef NVTX
           nvtxRangePop();
            nvtxRangePushA("put_the_stayed_agents_into_agentLocationAfterMovements");
#endif
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(//put the stayed agents into the agentLocationAfterMovements[i]
                structForAgents.begin(),
                thrust::make_permutation_iterator(agentLocationAfterMovement.begin(), stayedAngentsHelperVector.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                structForAgents.end(),
                thrust::make_permutation_iterator(agentLocationAfterMovement.begin(), stayedAngentsHelperVector.end()))),
                [rank, ITER, iter_exchange_number] __host__ __device__ ( thrust::tuple< thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>&,
                thrust::tuple<unsigned, unsigned,unsigned,unsigned, unsigned, unsigned>&> tup) {
                thrust::tuple<unsigned,unsigned,unsigned, unsigned, unsigned, unsigned> structForAgent =thrust::get<0>(tup);  
                unsigned iter_number_agent_go_to_destination = thrust::get<4>(structForAgent);
                unsigned iter_number_agent_go_home = thrust::get<5>(structForAgent);
                bool agent_going_destination_partition = (iter_number_agent_go_to_destination == (unsigned(ITER)));
                bool agent_going_home_partition = (iter_number_agent_go_home == (unsigned(ITER)));
                int difference_between_iterations_and_dest_time_dest = ITER - iter_number_agent_go_to_destination;
                int difference_between_iterations_and_dest_time_home = ITER - iter_number_agent_go_home;
                bool agent_will_go_dest_in_next_MPI_comm = difference_between_iterations_and_dest_time_dest > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_dest;
                bool agent_will_go_home_in_next_MPI_comm = difference_between_iterations_and_dest_time_home > 0 && iter_exchange_number >= difference_between_iterations_and_dest_time_home;
                if (!(agent_will_go_dest_in_next_MPI_comm || agent_will_go_home_in_next_MPI_comm)){//find out which agent will leave the city

                        thrust::get<1>(tup) = structForAgent;/*thrust::make_tuple<unsigned,unsigned,unsigned, unsigned, unsigned, unsigned>(thrust::get<0>(structForAgent),
                        thrust::get<1>(structForAgent),thrust::get<2>(structForAgent),thrust::get<3>(structForAgent),
                        thrust::get<4>(structForAgent),thrust::get<5>(structForAgent));*/
                    }
                    });
#ifdef NVTX
          nvtxRangePop();
          nvtxRangePushA("put_the_incoming_agents_into_agentLocationAfterMovements");  
#endif
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
            thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>* gpuptr2 = thrust::raw_pointer_cast(&IncomingAgent[0]);
            thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>* cpuptr2 = thrust::raw_pointer_cast(&hostIncomingAgent[0]);
            cudaMemcpy(gpuptr2,cpuptr2,IncomingAgent.size()*sizeof( thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>),cudaMemcpyHostToDevice);  
#else       
            thrust::copy(hostIncomingAgent.begin(), hostIncomingAgent.end(), IncomingAgent.begin());
#endif
                        
            
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(//this will put the incoming agents to the end of the agentLocationAfterMovements[i]
                agentLocationAfterMovement.begin()+vector_size-movedAgentSizeFromCity,
                IncomingAgent.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(
                agentLocationAfterMovement.end(),
                IncomingAgent.end()))
                ,[locationNumberPerCity] __host__ __device__ (thrust::tuple< thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>&,
                thrust::tuple<unsigned,unsigned,unsigned, unsigned, unsigned, unsigned>&> tup) {   
                    thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned> incoming_agent = thrust::get<1>(tup);
                    thrust::get<1>(incoming_agent) = RandomGenerator::randomUnsigned(locationNumberPerCity);
                    unsigned newLoc =  locationNumberPerCity-1;             
                    thrust::get<0>(tup)=incoming_agent;
            });
            auto t10 = std::chrono::high_resolution_clock::now(); 
            copying_agents_to_agentLocationAfterMovement += std::chrono::duration_cast<std::chrono::microseconds>(t10-t9).count();
#ifdef NVTX
        nvtxRangePop();
#endif

        }else{
            auto t11 = std::chrono::high_resolution_clock::now(); 
            agentLocationAfterMovement.resize(structForAgents.size());
            thrust::copy(structForAgents.begin(),structForAgents.end(),agentLocationAfterMovement.begin());
            auto t12 = std::chrono::high_resolution_clock::now(); 
            copying_agents_when_no_communication += std::chrono::duration_cast<std::chrono::microseconds>(t12-t11).count();
            
        }


        //this is only needed for printing
        auto t13 = std::chrono::high_resolution_clock::now(); 
        if(print_on){
            thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> hostagentLocationAfterMovement1(agentLocationAfterMovement.size());
            std::copy(agentLocationAfterMovement.begin(), agentLocationAfterMovement.end(), hostagentLocationAfterMovement1.begin());
            std::cout<<" agents in city "<< rank << " "<< agentLocationAfterMovement.size() << " at iter :"<< ITER << "\n";
            for(unsigned j=0;j<agentLocationAfterMovement.size();j++){
                auto id = thrust::get<0>(hostagentLocationAfterMovement1[j]);
                auto loc = thrust::get<1>(hostagentLocationAfterMovement1[j]);
                auto home = thrust::get<2>(hostagentLocationAfterMovement1[j]);
                auto dest = thrust::get<3>(hostagentLocationAfterMovement1[j]);
                auto goingiter = thrust::get<4>(hostagentLocationAfterMovement1[j]);
                auto comingiter = thrust::get<5>(hostagentLocationAfterMovement1[j]);
                std::cout << " ID " <<id << " loc " <<loc  << " home "<<  home << " dest " <<dest << " goingiter " <<goingiter  << " comingiter "<<  comingiter<<std::endl;  
            }
    
            std::cout<<"\n";
        }
        
        //prepare vectors for new loop
        //firstly the hostAgentLocation and the IDs:
        vector_size=agentLocationAfterMovement.size();
        agentID.resize(vector_size);
        hostAgentLocation.resize(vector_size);

#ifdef NVTX
        nvtxRangePushA("copy_the_needed_information_to_agentIDs_and_hostAgentLocations_arrays");
#endif
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                        agentID.begin(),
                        hostAgentLocation.begin(),
                        agentLocationAfterMovement.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(
                        agentID.end(),
                        hostAgentLocation.end(),
                        agentLocationAfterMovement.end())),
            [] __host__ __device__ (thrust::tuple<unsigned&, unsigned&, thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>&> tup) {
                thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned> movement = thrust::get<2>(tup);
                thrust::get<0>(tup) = thrust::get<0>(movement);
                thrust::get<1>(tup) = thrust::get<1>(movement);
            
            });
#ifdef NVTX
        nvtxRangePop();
#endif
#ifdef NVTX
        nvtxRangePushA("copy_IN_THE_END");
#endif
        if ((ITER+1)%iter_exchange_number==0){
            structForAgents.resize(vector_size);
            thrust::copy(agentLocationAfterMovement.begin(),agentLocationAfterMovement.end(),structForAgents.begin());
        }
#ifdef NVTX
        nvtxRangePop();
#endif            
        if(print_on){
            std::cout<<"ids and locations after movement"<<"\n";
            thrust::copy(agentID.begin(), agentID.end(), std::ostream_iterator<unsigned>(std::cout, "\t"));
            std::cout<<"\n";
            thrust::copy(hostAgentLocation.begin(), hostAgentLocation.end(), std::ostream_iterator<unsigned>(std::cout, "\t"));
            std::cout<<"\n";
            std::cout<<"\n \n";
        }          

    
        
        auto t14 = std::chrono::high_resolution_clock::now(); 
        create_the_new_arrays_after_movement += std::chrono::duration_cast<std::chrono::microseconds>(t14-t13).count();   
    }
    auto sum2 = std::chrono::high_resolution_clock::now();
    auto sumtime = std::chrono::duration_cast<std::chrono::microseconds>(sum2-sum1).count();
    
     int global_sum, global_update_arrays_time, global_movement_time, global_picking_out_stayed_exchanged_agents
     ,global_create_the_new_arrays_after_movement,global_exchanging_agents_with_mpi
     ,global_copying_agents_when_no_communication, global_sorting_and_picking_exchanging_agents
     , global_copying_agents_to_agentLocationAfterMovement  = 0;
     int min_exchanging_agents_with_mpi, max_exchanging_agents_with_mpi = 0;

    //int sum_time = (int) sumtime;
    MPI_Allreduce(&sumtime, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&update_arrays_time, &global_update_arrays_time, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&movement_time, &global_movement_time, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&picking_out_stayed_exchanged_agents, &global_picking_out_stayed_exchanged_agents, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&create_the_new_arrays_after_movement, &global_create_the_new_arrays_after_movement, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&exchanging_agents_with_mpi, &global_exchanging_agents_with_mpi, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&copying_agents_when_no_communication, &global_copying_agents_when_no_communication, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sorting_and_picking_exchanging_agents, &global_sorting_and_picking_exchanging_agents, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&copying_agents_to_agentLocationAfterMovement, &global_copying_agents_to_agentLocationAfterMovement, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&exchanging_agents_with_mpi, &min_exchanging_agents_with_mpi, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&exchanging_agents_with_mpi, &max_exchanging_agents_with_mpi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    global_sum = global_sum / size;
    global_update_arrays_time = global_update_arrays_time / size;
    global_movement_time = global_movement_time / size;
    global_picking_out_stayed_exchanged_agents = global_picking_out_stayed_exchanged_agents / size;
    global_create_the_new_arrays_after_movement = global_create_the_new_arrays_after_movement / size;
    global_exchanging_agents_with_mpi = global_exchanging_agents_with_mpi / size;
    global_copying_agents_when_no_communication = global_copying_agents_when_no_communication / size;
    global_sorting_and_picking_exchanging_agents = global_sorting_and_picking_exchanging_agents / size;
    global_copying_agents_to_agentLocationAfterMovement = global_copying_agents_to_agentLocationAfterMovement / size;
    if(rank==0 ){
        /*std::cout << "Setup_and_memory_resolution_took(microseconds), "<< std::chrono::duration_cast<std::chrono::microseconds>(t02-t01).count()<< "\n";
        std::cout<<"update_arrays_time(microseconds), "<<update_arrays_time<< "\n";
        std::cout<<"movement_time(microseconds), "<<movement_time<< "\n";
        std::cout<<"picking_out_stayed_exchanged_agents(microseconds), "<<picking_out_stayed_exchanged_agents<< "\n";
        std::cout<<"create_the_new_arrays_after_movement(microseconds), "<<create_the_new_arrays_after_movement<< "\n";
        std::cout<<"exchanging_agents_with_mpi(microseconds), "<<exchanging_agents_with_mpi<< "\n";
        std::cout<<"copying_agents_when_no_communication(microseconds), "<<copying_agents_when_no_communication<< "\n";
        std::cout<<"sorting_and_picking_exchanging_agents(microseconds), "<<sorting_and_picking_exchanging_agents<< "\n";
        std::cout<<"copying_agents_to_agentLocationAfterMovement(microseconds), "<<copying_agents_to_agentLocationAfterMovement<< "\n";
        std::cout<<"minimal communication time "<<min_exchanging_agents_with_mpi<< "\n";
        std::cout<<"maximal communication time "<<max_exchanging_agents_with_mpi<< "\n";


        std::cout<< "sum(microseconds), "<<  sumtime<<  "\n";*/




        
        std::cout <<"Average_Setup_and_memory_resolution_took(microseconds), "<< std::chrono::duration_cast<std::chrono::microseconds>(t02-t01).count()<< "\n";
        std::cout<<"Average_update_arrays_time(microseconds), "<<global_update_arrays_time<< "\n";
        std::cout<<"Average_movement_time(microseconds), "<<global_movement_time<< "\n";
        std::cout<<"Average_picking_out_stayed_exchanged_agents(microseconds), "<<global_picking_out_stayed_exchanged_agents<< "\n";
        std::cout<<"Average_create_the_new_arrays_after_movement(microseconds), "<<global_create_the_new_arrays_after_movement<< "\n";
        std::cout<<"Average_copying_agents_when_no_communication(microseconds), "<<global_copying_agents_when_no_communication<< "\n";
        std::cout<<"Average_sorting_and_picking_exchanging_agents(microseconds), "<<global_sorting_and_picking_exchanging_agents<< "\n";
        std::cout<<"Average_copying_agents_to_agentLocationAfterMovement(microseconds), "<<global_copying_agents_to_agentLocationAfterMovement<< "\n";
        std::cout<<"Average_exchanging_agents_with_mpi(microseconds), "<<global_exchanging_agents_with_mpi<< "\n";
        std::cout<<"minimal_communication_time, "<<min_exchanging_agents_with_mpi<< "\n";
        std::cout<<"maximal_communication_time, "<<max_exchanging_agents_with_mpi<< "\n";
        std::cout<<"Average_sum(microseconds), "<<  global_sum<<  "\n";
    }

    MPI_Finalize();
   


    delete[] requests;




}
PostMovement::PostMovement(unsigned NUM_OF_CITIES, int NUM_OF_ITERATIONS, double movedRatioInside, double movedRatioOutside,
 unsigned locations,unsigned print_on, unsigned rank, unsigned size,unsigned iter_exchange_number, unsigned NUM_OF_AGENTS,
 thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned >> structForAgents, thrust::device_vector<unsigned> agentID )
                : generatorHelper(NUM_OF_AGENTS/NUM_OF_CITIES*2.0)
                 {

    helperFunction(NUM_OF_CITIES,NUM_OF_ITERATIONS,movedRatioInside,movedRatioOutside,locations,print_on,
    generatorHelper,agentID,locationAgentList,hostAgentLocation,offset,placeToCopyAgentLength,hostMovement,hostexChangeAgent,
    exChangeAgent,offsetForExChangeAgent,movedAgentSizeFromCity,exchangeHelperVector,stayedAngentsHelperVector,IncomingAgent,hostIncomingAgent,
    agentLocationAfterMovement,rank,size, iter_exchange_number,NUM_OF_AGENTS, structForAgents);

    
}