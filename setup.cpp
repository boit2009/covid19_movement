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
//#include "nvToolsExt.h"
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


struct ZipComparator
{
    __host__ __device__
    inline bool operator() (const thrust::tuple<unsigned , unsigned, unsigned> &a, const thrust::tuple<unsigned, unsigned, unsigned> &b)
    {
        return thrust::get<2>(a) < thrust::get<2>(b);
    }
};
void helperFunction(unsigned NUM_OF_CITIES, int NUM_OF_ITERATIONS,unsigned agents, double movedRatioInside, double movedRatioOutside, unsigned locations,unsigned print_on,
    thrust::device_vector<unsigned> &generatorHelper,
    thrust::device_vector<unsigned> &agentID,
    thrust::device_vector<thrust::tuple<unsigned, unsigned>> &locationAgentList,
    thrust::device_vector<unsigned>&hostAgentLocation,
    thrust::device_vector<unsigned> &offset,
    thrust::device_vector<unsigned>&placeToCopyAgentLength,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned>> &hostMovement,
    thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned>> &hostexChangeAgent,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned>> &exChangeAgent,
    thrust::device_vector<unsigned> &offsetForExChangeAgent,
    unsigned &movedAgentSizeFromCity,
    thrust::device_vector<unsigned> &exchangeHelperVector,
    thrust::device_vector<unsigned> &stayedAngentsHelperVector,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned >>&IncomingAgent,
    thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned >>&hostIncomingAgent,
    thrust::device_vector<thrust::tuple<unsigned, unsigned,unsigned>>&agentLocationAfterMovement,
    unsigned rank,
    unsigned size,
    unsigned iter_exchange_number){
    
    MPI_Request *requests =new MPI_Request[NUM_OF_CITIES*2 - 2];


    RandomGenerator::init(agents);
    unsigned locationNumberPerCity = (locations / NUM_OF_CITIES) - 1; 
    if(print_on && rank == 0)
        printf("The_locationNumberPerCity_value_is:, %d \n", locationNumberPerCity);
    auto t01 = std::chrono::high_resolution_clock::now();
    //reserving memory 
    //nvtxRangePushA("init_host_data");
    locationAgentList.reserve(agents/NUM_OF_CITIES*1.2);
    hostAgentLocation.reserve(agents/NUM_OF_CITIES*1.2);
    offset.resize((locations/NUM_OF_CITIES) + 1);
    placeToCopyAgentLength.reserve(agents/NUM_OF_CITIES*1.2);
    hostMovement.reserve(agents/NUM_OF_CITIES*1.2);
    hostexChangeAgent.reserve(agents/NUM_OF_CITIES*1.2);
    exChangeAgent.reserve(agents/NUM_OF_CITIES*1.2);
    offsetForExChangeAgent.reserve(NUM_OF_CITIES+1);
    offsetForExChangeAgent.resize(NUM_OF_CITIES+1);
    exchangeHelperVector.reserve(agents/NUM_OF_CITIES*1.2);
    stayedAngentsHelperVector.reserve(agents/NUM_OF_CITIES*1.2);
    IncomingAgent.reserve(agents/NUM_OF_CITIES*1.2);
    hostIncomingAgent.reserve(agents/NUM_OF_CITIES*1.2);
    agentLocationAfterMovement.reserve(agents/NUM_OF_CITIES*1.2);
   // nvtxRangePop();
    
    generatorHelper.reserve(agents/NUM_OF_CITIES*1.2);
    unsigned vector_size;
    
    
    
    //firstly generate the locations and the agentIDs
    vector_size= agents/NUM_OF_CITIES;
    agentID.resize(vector_size);
   // nvtxRangePushA("genereate_sequences");
    thrust::sequence(agentID.begin(), agentID.end(), 0 + rank *  vector_size);
   // nvtxRangePop();
    hostAgentLocation.resize(vector_size);
  //  nvtxRangePushA("genereate_random locations");
    thrust::generate(hostAgentLocation.begin(), hostAgentLocation.end(), [locationNumberPerCity] __host__ __device__(){
        return RandomGenerator::randomUnsigned(locationNumberPerCity);
    });
   // nvtxRangePop();
    if(print_on){
        std::cout << "\n Generated locations\n ";
        thrust::copy(hostAgentLocation.begin(), hostAgentLocation.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        std::cout << "\n ";
    }
    hostMovement.resize(agents/NUM_OF_CITIES*1.2);
    thrust::for_each(hostMovement.begin(), hostMovement.end(),[rank] __host__ __device__ ( thrust::tuple<unsigned, unsigned, unsigned>& tup) {//fill the cityvalue in the beginning
                thrust::get<2>(tup)=rank;
                thrust::get<0>(tup)=rank; //to be inited
                thrust::get<1>(tup)=rank;  
        }); 
    hostMovement.resize(vector_size);


    
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
        std::cout<< "iter "<< ITER <<"\n";
        if(ITER == 0){
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
     //   nvtxRangePushA("copy_the_host_agent_locations_to_copy_array_in_the_beginning");             
        thrust::copy(hostAgentLocation.begin(), hostAgentLocation.end(), placeToCopyAgentLength.begin());
     //   nvtxRangePop();
     //nvtxRangePushA("sequence_generation_before_sorting_in_the_beginning"); 
        thrust::sequence(generatorHelper.begin(), generatorHelper.end());
    //nvtxRangePop();
    //nvtxRangePushA("sorting_arrays_in_the_beginning"); 
        thrust::stable_sort_by_key(placeToCopyAgentLength.begin(), placeToCopyAgentLength.end(), generatorHelper.begin());
    //    nvtxRangePop();
        if(print_on){
            std::cout << "Generate locationagentlist \n ";
            thrust::copy(generatorHelper.begin(), generatorHelper.end(), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << "\n";
            thrust::copy(placeToCopyAgentLength.begin(), placeToCopyAgentLength.end(), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << "\n";
        }
   //     nvtxRangePushA("sequence_generation_before_lower_bound_in_the_beginning"); 
        thrust::sequence(offset.begin(), offset.end());
    //    nvtxRangePop();
   //     nvtxRangePushA("lower_bound_time_in_the_beginning"); 
        thrust::lower_bound(placeToCopyAgentLength.begin(),placeToCopyAgentLength.end(),offset.begin(), offset.end(),offset.begin());
   //     nvtxRangePop();


        
        if (print_on){
            std::cout<<"\n";
            thrust::copy(offset.begin(), offset.end(), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout<<"\n";
        }

        
        auto t2 = std::chrono::high_resolution_clock::now();
        update_arrays_time += std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();


        
        // generate movement
        auto t3 = std::chrono::high_resolution_clock::now();
        
        agentID.resize(vector_size);
        exchangeHelperVector.resize(vector_size);
        stayedAngentsHelperVector.resize(vector_size);
    //    nvtxRangePushA("fill_helper_vectos_with_zeros"); 
        thrust::fill(exchangeHelperVector.begin(), exchangeHelperVector.end(), 0);
        thrust::fill(stayedAngentsHelperVector.begin(), stayedAngentsHelperVector.end(), 0);                       
   //    nvtxRangePop();
        hostMovement.resize(vector_size);
  //      nvtxRangePushA("generate_the_random_movement");
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(agentID.begin(), hostAgentLocation.begin(),hostMovement.begin()))
                        , thrust::make_zip_iterator(thrust::make_tuple(agentID.end(), hostAgentLocation.end(),hostMovement.end()))
                        , hostMovement.begin()
                        , [rank, movedRatioOutside, movedRatioInside, NUM_OF_CITIES, locationNumberPerCity, print_on]  __host__ __device__ (thrust::tuple<unsigned&, unsigned&,thrust::tuple<unsigned,unsigned,unsigned>&> idLocPair)  {
            unsigned idx = thrust::get<0>(idLocPair);
            unsigned loc = thrust::get<1>(idLocPair);
            thrust::tuple<unsigned, unsigned, unsigned> hostmovement = thrust::get<2>(idLocPair);
            double generatedRandom = RandomGenerator::randomUnit();
            unsigned old_loc = thrust::get<2>(hostmovement);
           // printf("id : %d is the old loc for %d \n", old_loc, idx);
            if(old_loc == rank){ //rakjam be a maradt agentet az adott varos utolso locationjera, amikor elmegy
             // amikor megérkezik, akkor random helyre rakjam
                if( generatedRandom < movedRatioOutside) { //if agent go to other city
                    if(print_on)
                        printf("id : %d goes to other city \n", idx);
                    loc = RandomGenerator::randomUnsigned(locationNumberPerCity);
                    unsigned where_to_go=rank;
                    if (NUM_OF_CITIES != 1)
                        while(where_to_go == rank) { where_to_go = 0 + ( RandomGenerator::randomUnsigned(NUM_OF_CITIES)); }
                    unsigned newLoc =  locationNumberPerCity-1; 
                    return thrust::make_tuple(idx ,newLoc, where_to_go);
                    }       
            
                else{
                    if(generatedRandom < movedRatioInside) { //if agent does not go to other city     
                        /* if(print_on)
                            std::cout <<" id " << idx << " goes inside " << std::endl;*/
                        unsigned newLoc =  RandomGenerator::randomUnsigned(locationNumberPerCity);
                        while(newLoc == loc) { newLoc =  RandomGenerator::randomUnsigned(locationNumberPerCity); }
                            return thrust::make_tuple(idx ,newLoc, rank);
                        }
                    return thrust::make_tuple(idx, loc, rank);
                }
            }else{
                if(print_on)
                    printf("loc : %d is the old loc for %d \n", old_loc, idx);
                return hostmovement;
            }
            

            
        });
        auto t4 = std::chrono::high_resolution_clock::now();
        movement_time += std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();
        if ((ITER+1)%iter_exchange_number==0){
        //    nvtxRangePop();
            
            auto t4_1 = std::chrono::high_resolution_clock::now(); 
                
            
            //  std::cout << "\n";   
            movedAgentSizeFromCity=0;
    //     nvtxRangePushA("fill_the_helper_vectors_based_on_the_movement");
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                                    hostMovement.begin(),
                                    exchangeHelperVector.begin(),
                                    stayedAngentsHelperVector.begin())),
                                    thrust::make_zip_iterator(thrust::make_tuple(
                                    hostMovement.end(),
                                    exchangeHelperVector.end(),
                                    stayedAngentsHelperVector.end())),
                                    [rank] __host__ __device__ ( thrust::tuple<thrust::tuple<unsigned, unsigned, unsigned>&, unsigned&, unsigned&> tup) {
                thrust::tuple<unsigned, unsigned, unsigned> hostmovement = thrust::get<0>(tup);                        
                unsigned city =thrust::get<2>(hostmovement);
                if (city != rank){//find out which agent will leave the city
                    thrust::get<1>(tup)=1;
                }else{//the others will stay
                    thrust::get<2>(tup)=1;
                }
            }); 
    //      nvtxRangePop();

        
            auto t4_2 = std::chrono::high_resolution_clock::now();
            picking_out_stayed_exchanged_agents += std::chrono::duration_cast<std::chrono::microseconds>(t4_2-t4_1).count(); 
            auto t5 = std::chrono::high_resolution_clock::now(); 
            
            
            unsigned exchangedAgents = exchangeHelperVector[hostMovement.size()-1];
    //     nvtxRangePushA("exclusive_scan_with_exchange_helper_vectors");
            thrust::exclusive_scan(exchangeHelperVector.begin(), exchangeHelperVector.end(), exchangeHelperVector.begin());
    //      nvtxRangePop();
            movedAgentSizeFromCity = exchangedAgents + exchangeHelperVector[hostMovement.size()-1]; //to know how many agent left city
            exChangeAgent.resize(movedAgentSizeFromCity);
            if(print_on)
                hostexChangeAgent.resize(movedAgentSizeFromCity);
    //     nvtxRangePushA("put_leaving_agents_into_exchangeagents");
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(// this for each will put the leaving agents into the exChangeAgents[i].
                        hostMovement.begin(),
                        thrust::make_permutation_iterator(exChangeAgent.begin(), exchangeHelperVector.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                        hostMovement.end(),
                        thrust::make_permutation_iterator(exChangeAgent.begin(), exchangeHelperVector.end()))),
                [rank] __host__ __device__ ( thrust::tuple<thrust::tuple<unsigned, unsigned,unsigned>&, thrust::tuple<unsigned, unsigned,unsigned>&> tup) {
                thrust::tuple<unsigned, unsigned,unsigned> hostmovement =thrust::get<0>(tup);
                unsigned city =thrust::get<2>(hostmovement);
                    if (city != rank){
                        unsigned id = thrust::get<0>(hostmovement);
                        unsigned loc = thrust::get<1>(hostmovement);  
                        thrust::get<1>(tup)= thrust::make_tuple<unsigned, unsigned, unsigned>(id,loc,city); //put the values to exchangeagents
                    }
            });
            //nvtxRangePop();
            if(print_on)
                hostexChangeAgent = exChangeAgent; // to be able to print, this can be removed later
        //    nvtxRangePushA("stable_sort_on_exchange_agnets");
            thrust::stable_sort(exChangeAgent.begin(),exChangeAgent.end(), ZipComparator()); //sort the exchanging array by city
    //     nvtxRangePop();
            thrust::device_vector<unsigned>cityIndex(exChangeAgent.size());
    //      nvtxRangePushA("pull_out_the_needed_city_indexes");
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                        cityIndex.begin(),
                        exChangeAgent.begin()))
                        ,thrust::make_zip_iterator(thrust::make_tuple(
                        cityIndex.end(),
                        exChangeAgent.end())),[] __host__ __device__ (thrust::tuple<unsigned&, thrust::tuple<unsigned, unsigned, unsigned>&> tup) {
                            thrust::tuple<unsigned, unsigned, unsigned> &movement =thrust::get<1>(tup);
                            thrust::get<0>(tup) = thrust::get<2>(movement);//put the city indexes into cityIndex array
                            
                });
                
    //     nvtxRangePop();
    //      nvtxRangePushA("sequence_before_offsetForExChangeAgents_lower_bound");
            thrust::sequence(offsetForExChangeAgent.begin(), offsetForExChangeAgent.end());
    //      nvtxRangePop();
    //      nvtxRangePushA("lower_bound_on_offsetForExChangeAgents");
            thrust::lower_bound(cityIndex.begin(),cityIndex.end(),offsetForExChangeAgent.begin(),
            offsetForExChangeAgent.end(),offsetForExChangeAgent.begin());
    //        nvtxRangePop();

        
            hostexChangeAgent.resize(exChangeAgent.size());
            thrust::copy(exChangeAgent.begin(), exChangeAgent.end(), hostexChangeAgent.begin());
            auto t6 = std::chrono::high_resolution_clock::now(); 
            sorting_and_picking_exchanging_agents += std::chrono::duration_cast<std::chrono::microseconds>(t6-t5).count(); 
            auto t7 = std::chrono::high_resolution_clock::now(); 
            unsigned counter=0;
            unsigned incomingAgentsNumberToParticularCity=0;
            unsigned *incoming_agents_array = new unsigned[NUM_OF_CITIES];//felszabadítani
            for(unsigned j = 0; j < NUM_OF_CITIES;j++){
                if(j !=rank){
                    unsigned number_of_sent_agents = offsetForExChangeAgent[j+1] - offsetForExChangeAgent[j]; 
                    MPI_Isend(&number_of_sent_agents, 1, MPI_UNSIGNED, j, rank, MPI_COMM_WORLD, &requests[2*counter + 0]);
                    MPI_Irecv(&incoming_agents_array[j], 1, MPI_UNSIGNED, j, j, MPI_COMM_WORLD, &requests[2*counter + 1]);
                    counter++;
                }
            }
            MPI_Waitall(2*NUM_OF_CITIES - 2, &requests[0], MPI_STATUSES_IGNORE);
            for(unsigned j = 0; j < NUM_OF_CITIES;j++){
                if(j !=rank){
                    incomingAgentsNumberToParticularCity +=incoming_agents_array[j];
                }
            }
            
            hostIncomingAgent.resize(incomingAgentsNumberToParticularCity);
            counter=0;
            for(unsigned j = 0; j < NUM_OF_CITIES;j++){
                if(j !=rank){
                    unsigned number_of_sent_agents = offsetForExChangeAgent[j+1] - offsetForExChangeAgent[j];
                    MPI_Isend((unsigned*)&hostexChangeAgent[offsetForExChangeAgent[j]], 3*number_of_sent_agents, MPI_UNSIGNED, j, rank, MPI_COMM_WORLD, &requests[2*counter + 0]);
                    unsigned where_to_start=0;
                    for(int i=0;i<j;i++){
                        if(i!=rank){
                            where_to_start+= incoming_agents_array[i];
                        }
                    }
                    MPI_Irecv(&hostIncomingAgent[where_to_start],3*incoming_agents_array[j], MPI_UNSIGNED, j, j, MPI_COMM_WORLD, &requests[2*counter + 1]);
                    counter++;
                }
            }
            MPI_Waitall(2*NUM_OF_CITIES - 2, &requests[0], MPI_STATUSES_IGNORE); 
            delete[] incoming_agents_array;
            auto t8 = std::chrono::high_resolution_clock::now(); 
            exchanging_agents_with_mpi += std::chrono::duration_cast<std::chrono::microseconds>(t8-t7).count();
            auto t9 = std::chrono::high_resolution_clock::now(); 
        
        
            if(print_on){
                    hostexChangeAgent = exChangeAgent; // to be able to print, this can be removed later
                std::cout << "After sorting: Agents moving from "<< rank+1 << "st city :"<< movedAgentSizeFromCity <<"\n ";
                for(unsigned  j =0;j<  hostexChangeAgent.size();j++){
                    unsigned id =thrust::get<0>(hostexChangeAgent[j]);
                    unsigned city =thrust::get<2>(hostexChangeAgent[j]);
                    unsigned loc =thrust::get<1>(hostexChangeAgent[j]);
                    std::cout<<  " ID" <<id << " city " << city<< " loc " << loc;
                }
                std::cout << "\n";
            }
            // in the end we have the exchanging arrays and the exchanging offset array
    
            
            if(print_on)
                std::cout<<"\n";

        
            //generate the locations after the movements
            if(print_on)
                std::cout<<"incomingAgentsNumberToParticularCity "<<incomingAgentsNumberToParticularCity<<"\n";

            IncomingAgent.resize(incomingAgentsNumberToParticularCity);
            //the new size will be the original size - left agents size + incoming agents size
            agentLocationAfterMovement.resize(vector_size-movedAgentSizeFromCity+incomingAgentsNumberToParticularCity);
    //      nvtxRangePushA("exclusive_scan_on_stayed_helper_vector");
            thrust::exclusive_scan(stayedAngentsHelperVector.begin(), stayedAngentsHelperVector.end(), stayedAngentsHelperVector.begin());
    //       nvtxRangePop();
    //        nvtxRangePushA("put_the_stayed_agents_into_agentLocationAfterMovements");
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(//put the stayed agents into the agentLocationAfterMovements[i]
                hostMovement.begin(),
                thrust::make_permutation_iterator(agentLocationAfterMovement.begin(), stayedAngentsHelperVector.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                hostMovement.end(),
                thrust::make_permutation_iterator(agentLocationAfterMovement.begin(), stayedAngentsHelperVector.end()))),
                [rank] __host__ __device__ ( thrust::tuple< thrust::tuple<unsigned, unsigned, unsigned>&,thrust::tuple<unsigned, unsigned,unsigned>&> tup) {
                    thrust::tuple<unsigned,unsigned,unsigned> hostmovement =thrust::get<0>(tup);  
                    unsigned city =thrust::get<2>(hostmovement); 
                    if (city == rank){
                        unsigned id = thrust::get<0>(hostmovement);
                        unsigned loc = thrust::get<1>(hostmovement); 
                        thrust::get<1>(tup)= thrust::make_tuple<unsigned,unsigned,unsigned>(id,loc,city);
                    }
                    });
    //      nvtxRangePop();
    //      nvtxRangePushA("put_the_incoming_agents_into_agentLocationAfterMovements");  
            thrust::copy(hostIncomingAgent.begin(),hostIncomingAgent.end(),IncomingAgent.begin()); //vissza kell ezt host vectorra másolni egyáltalán? !!!
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(//this will put the incoming agents to the end of the agentLocationAfterMovements[i]
                agentLocationAfterMovement.begin()+vector_size-movedAgentSizeFromCity,
                IncomingAgent.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(
                agentLocationAfterMovement.end(),
                IncomingAgent.end()))
                ,[locationNumberPerCity] __host__ __device__ (thrust::tuple< thrust::tuple<unsigned, unsigned, unsigned>&,thrust::tuple<unsigned,unsigned,unsigned>&> tup) {   
                    thrust::tuple<unsigned, unsigned, unsigned> incoming_agent = thrust::get<1>(tup);
                    thrust::get<1>(incoming_agent) = RandomGenerator::randomUnsigned(locationNumberPerCity);
                    unsigned newLoc =  locationNumberPerCity-1;             
                    thrust::get<0>(tup)=incoming_agent;
            });
            auto t10 = std::chrono::high_resolution_clock::now(); 
            copying_agents_to_agentLocationAfterMovement += std::chrono::duration_cast<std::chrono::microseconds>(t10-t9).count();
        }else{
            auto t11 = std::chrono::high_resolution_clock::now(); 
            agentLocationAfterMovement.resize(hostMovement.size());
            thrust::copy(hostMovement.begin(),hostMovement.end(),agentLocationAfterMovement.begin());
            auto t12 = std::chrono::high_resolution_clock::now(); 
            copying_agents_when_no_communication += std::chrono::duration_cast<std::chrono::microseconds>(t12-t11).count();
            
        }
   //     nvtxRangePop();


        //this is only needed for printing
        auto t13 = std::chrono::high_resolution_clock::now(); 
        if(print_on){
            thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned>> hostagentLocationAfterMovement1(agentLocationAfterMovement.size());
            std::copy(agentLocationAfterMovement.begin(), agentLocationAfterMovement.end(), hostagentLocationAfterMovement1.begin());
            std::cout<<" agents in city "<< rank << " "<< agentLocationAfterMovement.size() << " at iter :"<< ITER << "\n";
            for(unsigned j=0;j<agentLocationAfterMovement.size();j++){
                auto id = thrust::get<0>(hostagentLocationAfterMovement1[j]);
                auto loc = thrust::get<1>(hostagentLocationAfterMovement1[j]);
                auto city = thrust::get<2>(hostagentLocationAfterMovement1[j]);
                std::cout << " ID " <<id << " city " <<city  << " loc "<<  loc;  
            }
    
            std::cout<<"\n";
        }
        
        //prepare vectors for new loop
        //firstly the hostAgentLocation and the IDs:
        vector_size=agentLocationAfterMovement.size();
        agentID.resize(vector_size);
        hostAgentLocation.resize(vector_size);

    //    nvtxRangePushA("copy_the_needed_information_to_agentIDs_and_hostAgentLocations_arrays");
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                        agentID.begin(),
                        hostAgentLocation.begin(),
                        agentLocationAfterMovement.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(
                        agentID.end(),
                        hostAgentLocation.end(),
                        agentLocationAfterMovement.end())),
            [] __host__ __device__ (thrust::tuple<unsigned&, unsigned&, thrust::tuple<unsigned, unsigned, unsigned>&> tup) {
                thrust::tuple<unsigned, unsigned, unsigned> movement =thrust::get<2>(tup);
                thrust::get<0>(tup) = thrust::get<0>(movement);
                thrust::get<1>(tup) = thrust::get<1>(movement);
            
            });
   //     nvtxRangePop();
        if ((ITER+1)%iter_exchange_number==0){
            hostMovement.resize(vector_size);
            thrust::copy(agentLocationAfterMovement.begin(),agentLocationAfterMovement.end(),hostMovement.begin());
        }
            
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
    MPI_Finalize();
    if (print_on){
        std::cout << "Setup_and_memory_resolution_took(microseconds), "<< std::chrono::duration_cast<std::chrono::microseconds>(t02-t01).count()<< "\n";
        std::cout<<"update_arrays_time(microseconds), "<<update_arrays_time<< "\n";
        std::cout<<"movement_time(microseconds), "<<movement_time<< "\n";
        std::cout<<"picking_out_stayed_exchanged_agents(microseconds), "<<picking_out_stayed_exchanged_agents<< "\n";
        std::cout<<"create_the_new_arrays_after_movement(microseconds), "<<create_the_new_arrays_after_movement<< "\n";
        std::cout<<"exchanging_agents_with_mpi(microseconds), "<<exchanging_agents_with_mpi<< "\n";
        std::cout<<"copying_agents_when_no_communication(microseconds), "<<copying_agents_when_no_communication<< "\n";
        std::cout<<"sorting_and_picking_exchanging_agents(microseconds), "<<sorting_and_picking_exchanging_agents<< "\n";
        std::cout<<"copying_agents_to_agentLocationAfterMovement(microseconds), "<<copying_agents_to_agentLocationAfterMovement<< "\n";
        std::cout<< "sum(microseconds), "<<  sumtime<<  "\n";
    }
    else{
        if(rank==0){
            std::cout << "Setup_and_memory_resolution_took(microseconds), "<< std::chrono::duration_cast<std::chrono::microseconds>(t02-t01).count()<< "\n";
            std::cout<<"update_arrays_time(microseconds), "<<update_arrays_time<< "\n";
            std::cout<<"movement_time(microseconds), "<<movement_time<< "\n";
            std::cout<<"picking_out_stayed_exchanged_agents(microseconds), "<<picking_out_stayed_exchanged_agents<< "\n";
            std::cout<<"create_the_new_arrays_after_movement(microseconds), "<<create_the_new_arrays_after_movement<< "\n";
            std::cout<<"exchanging_agents_with_mpi(microseconds), "<<exchanging_agents_with_mpi<< "\n";
            std::cout<<"copying_agents_when_no_communication(microseconds), "<<copying_agents_when_no_communication<< "\n";
            std::cout<<"sorting_and_picking_exchanging_agents(microseconds), "<<sorting_and_picking_exchanging_agents<< "\n";
            std::cout<<"copying_agents_to_agentLocationAfterMovement(microseconds), "<<copying_agents_to_agentLocationAfterMovement<< "\n";
            std::cout<< "sum(microseconds), "<<  sumtime<<  "\n";
        }

    }
   


    delete[] requests;




}
PostMovement::PostMovement(unsigned NUM_OF_CITIES, int NUM_OF_ITERATIONS,unsigned agents, double movedRatioInside, double movedRatioOutside,
 unsigned locations,unsigned print_on, unsigned rank, unsigned size,unsigned iter_exchange_number)
                : generatorHelper(agents/NUM_OF_CITIES*1.2)
                 {

    helperFunction(NUM_OF_CITIES,NUM_OF_ITERATIONS,agents,movedRatioInside,movedRatioOutside,locations,print_on,
    generatorHelper,agentID,locationAgentList,hostAgentLocation,offset,placeToCopyAgentLength,hostMovement,hostexChangeAgent,
    exChangeAgent,offsetForExChangeAgent,movedAgentSizeFromCity,exchangeHelperVector,stayedAngentsHelperVector,IncomingAgent,hostIncomingAgent,
    agentLocationAfterMovement,rank,size, iter_exchange_number);

    
}

