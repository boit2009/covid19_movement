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
/*#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif*/

        struct ZipComparator
{
    __host__ __device__
    inline bool operator() (const thrust::tuple<unsigned , unsigned, unsigned> &a, const thrust::tuple<unsigned, unsigned, unsigned> &b)
    {
        return thrust::get<2>(a) < thrust::get<2>(b);
    }
};
PostMovement::PostMovement(unsigned NUM_OF_CITIES, unsigned NUM_OF_ITERATIONS,unsigned agents, double movedRatioInside, double movedRatioOutside, unsigned locations,unsigned print_on)
                : generatorHelper(agents/NUM_OF_CITIES*1.2)
                , agentIDs(NUM_OF_CITIES)
                ,locationAgentLists(NUM_OF_CITIES)
                ,hostAgentLocations(NUM_OF_CITIES)
                ,offsets(NUM_OF_CITIES)
                ,placeToCopyAgentLengths(NUM_OF_CITIES)
                ,hostMovements(NUM_OF_CITIES)
                ,hostexChangeAgents(NUM_OF_CITIES)
                ,exChangeAgents(NUM_OF_CITIES)
                ,offsetForExChangeAgents(NUM_OF_CITIES)
                ,movedAgentSizeFromCities(NUM_OF_CITIES)
                ,exchangeHelperVectors(NUM_OF_CITIES)
                ,stayedAngentsHelperVectors(NUM_OF_CITIES)
                ,IncomingAgents(NUM_OF_CITIES)
                ,hostIncomingAgents(NUM_OF_CITIES)
                ,agentLocationAfterMovements(NUM_OF_CITIES) {
    // generate agentLocationBeforeMovement
    // std::cout << "generate agentLocationBeforeMovement" << std::endl;
    std::random_device rd{};
    std::mt19937_64 generator{1};
    //srand((unsigned) time(0));
    std::uniform_int_distribution<unsigned> randomLocation{0, (locations / NUM_OF_CITIES) - 1};
    auto getRandomLocation = [&] __host__ __device__() -> unsigned { return randomLocation(generator); };
    std::uniform_real_distribution<decltype(movedRatioOutside)> randomUnit{0.0, 1.0}; 
    auto t01 = std::chrono::high_resolution_clock::now();
    for(unsigned i=0;i<NUM_OF_CITIES;i++){//reserving memory 
        locationAgentLists[i].reserve(agents/NUM_OF_CITIES*1.2);
        hostAgentLocations[i].reserve(agents/NUM_OF_CITIES*1.2);
        offsets[i].resize((locations/NUM_OF_CITIES) + 1);
        placeToCopyAgentLengths[i].reserve(agents/NUM_OF_CITIES*1.2);
        hostMovements[i].reserve(agents/NUM_OF_CITIES*1.2);
        hostexChangeAgents[i].reserve(agents/NUM_OF_CITIES*1.2);
        exChangeAgents[i].reserve(agents/NUM_OF_CITIES*1.2);
        offsetForExChangeAgents[i].reserve(NUM_OF_CITIES+1);
        offsetForExChangeAgents[i].resize(NUM_OF_CITIES+1);
        exchangeHelperVectors[i].reserve(agents/NUM_OF_CITIES*1.2);
        stayedAngentsHelperVectors[i].reserve(agents/NUM_OF_CITIES*1.2);
        IncomingAgents[i].reserve(agents/NUM_OF_CITIES*1.2);
        hostIncomingAgents[i].reserve(agents/NUM_OF_CITIES*1.2);
        agentLocationAfterMovements[i].reserve(agents/NUM_OF_CITIES*1.2);
    }
    generatorHelper.reserve(agents/NUM_OF_CITIES*1.2);
    std::vector<unsigned> vector_sizes(NUM_OF_CITIES);
    
    
    for(unsigned i=0;i<NUM_OF_CITIES;i++){//firstly generate the locations and the agentIDs
        vector_sizes[i]= agents/NUM_OF_CITIES;
        agentIDs[i].resize(vector_sizes[i]);
        thrust::sequence(agentIDs[i].begin(), agentIDs[i].end(), 0 + i *  vector_sizes[i]);
        hostAgentLocations[i].resize(vector_sizes[i]);
        thrust::generate(hostAgentLocations[i].begin(), hostAgentLocations[i].end(), getRandomLocation);
        if(print_on){
            std::cout << "\n Generated locations\n ";
            thrust::copy(hostAgentLocations[i].begin(), hostAgentLocations[i].end(), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << "\n ";
        }

    }
    auto t02 = std::chrono::high_resolution_clock::now();
    std::cout << "Setup and memory resolution took "<< std::chrono::duration_cast<std::chrono::microseconds>(t02-t01).count()<< " microseconds\n";
    
    auto update_arrays_time = 0;
    auto movement_time = 0;
    auto sorting_merging_arrays_after_movement = 0;
    auto copy_incoming_agents_and_create_the_new_arrays_after_movement = 0;


    for(unsigned ITER=0;ITER<NUM_OF_ITERATIONS;ITER++){

        auto t1 = std::chrono::high_resolution_clock::now();
        for(unsigned i=0;i<NUM_OF_CITIES;i++){
            placeToCopyAgentLengths[i].resize(vector_sizes[i]);
            locationAgentLists[i].resize(vector_sizes[i]);
            generatorHelper.resize(vector_sizes[i]);
            hostAgentLocations[i].resize(vector_sizes[i]);                  
            std::copy(hostAgentLocations[i].begin(), hostAgentLocations[i].end(), placeToCopyAgentLengths[i].begin());
            thrust::sequence(generatorHelper.begin(), generatorHelper.end());
            thrust::stable_sort_by_key(placeToCopyAgentLengths[i].begin(), placeToCopyAgentLengths[i].end(), generatorHelper.begin());
            if(print_on){
                std::cout << "Generate locationagentlist \n ";
                thrust::copy(generatorHelper.begin(), generatorHelper.end(), std::ostream_iterator<unsigned>(std::cout, " "));
                std::cout << "\n";
                thrust::copy(placeToCopyAgentLengths[i].begin(), placeToCopyAgentLengths[i].end(), std::ostream_iterator<unsigned>(std::cout, " "));
                std::cout << "\n";
            }

            thrust::sequence(offsets[i].begin(), offsets[i].end());
            thrust::lower_bound(placeToCopyAgentLengths[i].begin(),placeToCopyAgentLengths[i].end(),offsets[i].begin(), offsets[i].end(),offsets[i].begin());

           
             if (print_on){
                std::cout<<"\n";
                thrust::copy(offsets[i].begin(), offsets[i].end(), std::ostream_iterator<unsigned>(std::cout, " "));
                std::cout<<"\n";
             }

        }
        auto t2 = std::chrono::high_resolution_clock::now();
        update_arrays_time += std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();


        
        // generate movement
        auto t3 = std::chrono::high_resolution_clock::now();
        for(unsigned i=0;i<NUM_OF_CITIES;i++){
            agentIDs[i].resize(vector_sizes[i]);
            exchangeHelperVectors[i].resize(vector_sizes[i]);
            stayedAngentsHelperVectors[i].resize(vector_sizes[i]);
            thrust::for_each(exchangeHelperVectors[i].begin(), exchangeHelperVectors[i].end(),[&] __host__ __device__ (auto &v) {
                v=0;
            });
            thrust::for_each(stayedAngentsHelperVectors[i].begin(), stayedAngentsHelperVectors[i].end(),[&] __host__ __device__ (auto &v) {
                v=0;
            });
            hostMovements[i].resize(vector_sizes[i]);
            thrust::transform(thrust::make_zip_iterator(agentIDs[i].begin(), hostAgentLocations[i].begin())
                            , thrust::make_zip_iterator(agentIDs[i].end(), hostAgentLocations[i].end())
                            , hostMovements[i].begin()
                            , [&]  __host__ __device__ (const auto& idLocPair)  {
                auto idx = thrust::get<0>(idLocPair);
                auto loc = thrust::get<1>(idLocPair);
                auto generatedRandom = randomUnit(generator);

                if( generatedRandom < movedRatioOutside) { //if agent go to other city
                        // std::cout <<" id " << idx << " goes another city" << std::endl;
                        loc = getRandomLocation();
                        unsigned where_to_go=i;
                        if (NUM_OF_CITIES != 1)
                            while(where_to_go == i) { where_to_go = 0 + (rand() % NUM_OF_CITIES); }
                        auto newLoc = getRandomLocation();
                        while(newLoc == loc) { newLoc = getRandomLocation(); }
                        return thrust::make_tuple(idx ,newLoc, where_to_go);
                    }       
                
                else{
                    if(generatedRandom < movedRatioInside) { //if agent does not go to other city     
                        if(print_on)
                            std::cout <<" id " << idx << " goes inside " << std::endl;
                        auto newLoc = getRandomLocation();
                        while(newLoc == loc) { newLoc = getRandomLocation(); }
                            return thrust::make_tuple(idx ,newLoc, i);
                        }
                    return thrust::make_tuple(idx, loc, i);
                }
            });
            //  std::cout << "\n";   
            movedAgentSizeFromCities[i]=0;
            thrust::for_each(thrust::make_zip_iterator(
                                    hostMovements[i].begin(),
                                    exchangeHelperVectors[i].begin(),
                                    stayedAngentsHelperVectors[i].begin()),
                                    thrust::make_zip_iterator(
                                    hostMovements[i].end(),
                                    exchangeHelperVectors[i].end()
                                    ,stayedAngentsHelperVectors[i].end())
                                    ,[&] __host__ __device__ (const auto &tup) {
                auto hostmovement =thrust::get<0>(tup);                        
                auto city =thrust::get<2>(hostmovement);
                if (city != i){//find out which agent will leave the city
                    thrust::get<1>(tup)=1;
                }else{//the others will stay
                    thrust::get<2>(tup)=1;
                }
            }); 
                        
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        movement_time += std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count(); 
        auto t5 = std::chrono::high_resolution_clock::now();        
        for(unsigned i=0;i<NUM_OF_CITIES;i++){
            unsigned exchangedAgents= exchangeHelperVectors[i][hostMovements[i].size()-1];
            thrust::exclusive_scan(exchangeHelperVectors[i].begin(), exchangeHelperVectors[i].end(), exchangeHelperVectors[i].begin());
            movedAgentSizeFromCities[i] = exchangedAgents + exchangeHelperVectors[i][hostMovements[i].size()-1]; //to know how many agent left city
            exChangeAgents[i].resize(movedAgentSizeFromCities[i]);
            if(print_on)
                hostexChangeAgents[i].resize(movedAgentSizeFromCities[i]);
            thrust::for_each(thrust::make_zip_iterator(// this for each will put the leaving agents into the exChangeAgents[i].
                        hostMovements[i].begin(),
                        thrust::make_permutation_iterator(exChangeAgents[i].begin(), exchangeHelperVectors[i].begin())),
                thrust::make_zip_iterator(
                        hostMovements[i].end(),
                        thrust::make_permutation_iterator(exChangeAgents[i].begin(), exchangeHelperVectors[i].end())),
                [&] __host__ __device__ (const auto &tup) {
                auto hostmovement =thrust::get<0>(tup);
                auto city =thrust::get<2>(hostmovement);
                    if (city != i){
                        auto id = thrust::get<0>(hostmovement);
                        auto loc = thrust::get<1>(hostmovement);  
                        thrust::get<1>(tup)= thrust::make_tuple<unsigned, unsigned,unsigned>(id,loc,city); //put the values to exchangeagents
                        thrust::get<0>(tup) = thrust::make_tuple(UINT_MAX ,UINT_MAX, UINT_MAX); // zero the values in the original hostMovement array
                    }
                });
            if(print_on)
                hostexChangeAgents[i] = exChangeAgents[i]; // to be able to print, this can be removed later

            thrust::stable_sort(exChangeAgents[i].begin(),exChangeAgents[i].end(), ZipComparator()); //sort the exchanging array by city
            thrust::device_vector<unsigned>cityIndex(exChangeAgents[i].size());
            thrust::for_each(thrust::make_zip_iterator(
                        cityIndex.begin(),
                        exChangeAgents[i].begin())
                        ,thrust::make_zip_iterator(
                        cityIndex.end(),
                        exChangeAgents[i].end()),[&] __host__ __device__ (const auto &tup) {
                            auto movement =thrust::get<1>(tup);
                            thrust::get<0>(tup) = thrust::get<2>(movement);//put the city indexes into cityIndex array
                            
                });

            thrust::sequence(offsetForExChangeAgents[i].begin(), offsetForExChangeAgents[i].end());
            thrust::lower_bound(cityIndex.begin(),cityIndex.end(),offsetForExChangeAgents[i].begin(),
            offsetForExChangeAgents[i].end(),offsetForExChangeAgents[i].begin());
            
            if(print_on){
                 hostexChangeAgents[i] = exChangeAgents[i]; // to be able to print, this can be removed later
                std::cout << "After sorting: Agents moving from "<< i+1 << "st city :"<< movedAgentSizeFromCities[i] <<"\n ";
                for(unsigned  j =0;j<  hostexChangeAgents[i].size();j++){
                    std::cout<<  " ID" <<thrust::get<0>(hostexChangeAgents[i][j]) << " city " << thrust::get<2>(hostexChangeAgents[i][j])<< " loc " << thrust::get<1>(hostexChangeAgents[i][j]);
                }
                std::cout << "\n";
            }
        } // in the end we have the exchanging arrays and the exchanging offset array
        auto t6 = std::chrono::high_resolution_clock::now(); 
        sorting_merging_arrays_after_movement += std::chrono::duration_cast<std::chrono::microseconds>(t6-t5).count();   
        auto t7 = std::chrono::high_resolution_clock::now(); 
        if(print_on)
            std::cout<<"\n";

    
        for(unsigned i=0;i<NUM_OF_CITIES;i++){//generate the locations after the movements
            
            unsigned incomingAgentsNumberToParticularCity=0;
            for (unsigned city = 0;city<NUM_OF_CITIES;city++){
                unsigned from = offsetForExChangeAgents[city][i];
                unsigned to = offsetForExChangeAgents[city][i+1];
                incomingAgentsNumberToParticularCity+=(to-from);
                
            }
            if(print_on)
                std::cout<<"incomingAgentsNumberToParticularCity "<<incomingAgentsNumberToParticularCity<<"\n";

            IncomingAgents[i].resize(incomingAgentsNumberToParticularCity);
            unsigned numberOfAgentsPutToIncomingAgents=0;
            for (unsigned city =0;city<NUM_OF_CITIES;city++){// copy IncomingAgents 
                unsigned from =offsetForExChangeAgents[city][i];
                unsigned to=offsetForExChangeAgents[city][i+1];
                if(print_on)
                    std::cout<<"from city "<<city<<" to city "<< i <<" start: "<<from <<" end : "<<to<<"\n";
                thrust::for_each(thrust::make_zip_iterator(
                                exChangeAgents[city].begin()+from,
                                IncomingAgents[i].begin()+numberOfAgentsPutToIncomingAgents),
                                thrust::make_zip_iterator(
                                exChangeAgents[city].begin()+to,
                                IncomingAgents[i].begin()+numberOfAgentsPutToIncomingAgents+to-from)
                                ,[&] __host__ __device__ (const auto &tup) {                     
                            thrust::get<1>(tup)=thrust::get<0>(tup);
                            
                        });

                numberOfAgentsPutToIncomingAgents+=to-from;

            
            }


             //the new size will be the original size - left agents size + incoming agents size
            agentLocationAfterMovements[i].resize(vector_sizes[i]-movedAgentSizeFromCities[i]+incomingAgentsNumberToParticularCity);
            thrust::exclusive_scan(stayedAngentsHelperVectors[i].begin(), stayedAngentsHelperVectors[i].end(), stayedAngentsHelperVectors[i].begin());
            thrust::for_each(thrust::make_zip_iterator(//put the stayed agents into the agentLocationAfterMovements[i]
                            hostMovements[i].begin(),
                            thrust::make_permutation_iterator(agentLocationAfterMovements[i].begin(), stayedAngentsHelperVectors[i].begin())),
                            thrust::make_zip_iterator(
                            hostMovements[i].end(),
                            thrust::make_permutation_iterator(agentLocationAfterMovements[i].begin(), stayedAngentsHelperVectors[i].end())),
                    [&] __host__ __device__ (const auto &tup) {
                        auto hostmovement =thrust::get<0>(tup);  
                        auto id = thrust::get<0>(hostmovement);
                        auto loc = thrust::get<1>(hostmovement); 
                        auto city =thrust::get<2>(hostmovement); 
                        thrust::get<1>(tup)= thrust::make_tuple<unsigned, unsigned,unsigned>(id,loc,city);
        
                    });
                    
            thrust::for_each(thrust::make_zip_iterator(//this will put the incoming agents to the end of the agentLocationAfterMovements[i]
                agentLocationAfterMovements[i].begin()+vector_sizes[i]-movedAgentSizeFromCities[i],
                IncomingAgents[i].begin()),
                thrust::make_zip_iterator(
                agentLocationAfterMovements[i].end(),
                IncomingAgents[i].end())
                ,[&] __host__ __device__ (const auto &tup) {                     
                    thrust::get<0>(tup)=thrust::get<1>(tup);
            });


            //this is only needed for printing
            if(print_on){
                thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned>> hostagentLocationAfterMovement1(agentLocationAfterMovements[i].size());
                std::copy(agentLocationAfterMovements[i].begin(), agentLocationAfterMovements[i].end(), hostagentLocationAfterMovement1.begin());
                for(unsigned j=0;j<agentLocationAfterMovements[i].size()/* vector_sizes[i]-movedAgentSizeFromCities[i]+incomingAgentsNumberToParticularCity*/;j++){
                    auto id = thrust::get<0>(hostagentLocationAfterMovement1[j]);
                    auto loc = thrust::get<1>(hostagentLocationAfterMovement1[j]);
                    std::cout << " ID " <<id << "  loc "<<  loc;  
                }
            //  std::cout<<"\n currently in city"<<i<<" "<< agentLocationAfterMovements[i].size()<<"\n";
                std::cout<<"\n";
            }
            
            //prepare vectors for new loop
            //firstly the hostAgentLocation and the IDs:
            vector_sizes[i]=agentLocationAfterMovements[i].size();
            agentIDs[i].resize(vector_sizes[i]);
            hostAgentLocations[i].resize(vector_sizes[i]);
            thrust::for_each(thrust::make_zip_iterator(
                            agentIDs[i].begin(),
                            hostAgentLocations[i].begin(),
                            agentLocationAfterMovements[i].begin()),
                            thrust::make_zip_iterator(
                            agentIDs[i].end(),
                            hostAgentLocations[i].end()
                            ,agentLocationAfterMovements[i].end()),
                [&] __host__ __device__ (const auto &tup) {
                    auto movement =thrust::get<2>(tup);
                    thrust::get<0>(tup) = thrust::get<0>(movement);
                    thrust::get<1>(tup) = thrust::get<1>(movement);
                
                });
            if(print_on){
                std::cout<<"ids and locations after movement"<<"\n";
                thrust::copy(agentIDs[i].begin(), agentIDs[i].end(), std::ostream_iterator<unsigned>(std::cout, "\t"));
                std::cout<<"\n";
                thrust::copy(hostAgentLocations[i].begin(), hostAgentLocations[i].end(), std::ostream_iterator<unsigned>(std::cout, "\t"));
                std::cout<<"\n";
                std::cout<<"\n \n";
            }          

        
        }
            auto t8 = std::chrono::high_resolution_clock::now(); 
        copy_incoming_agents_and_create_the_new_arrays_after_movement += std::chrono::duration_cast<std::chrono::microseconds>(t8-t7).count();   
    }
    
    std::cout<<"update_arrays_time took "<<update_arrays_time<< " microseconds\n";
    std::cout<<"movement_time took "<<movement_time<< " microseconds\n";
    std::cout<<"sorting_merging_arrays_after_movement took "<<sorting_merging_arrays_after_movement/1000<< " microseconds\n";
    std::cout<<"copy_incoming_agents_and_create_the_new_arrays_after_movement took "<<copy_incoming_agents_and_create_the_new_arrays_after_movement<< " microseconds\n";


}

