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

    /*void initHelper1(thrust::device_vector<unsigned>& placeToCopyAgentLengths, thrust::device_vector<unsigned>& agentLocationBeforeMovement, thrust::device_vector<thrust::pair<unsigned, unsigned>>& agentListWithPairs) {
        thrust::transform(thrust::make_zip_iterator(placeToCopyAgentLengths.begin(), agentLocationBeforeMovement.begin())
                            , thrust::make_zip_iterator(placeToCopyAgentLengths.end(), agentLocationBeforeMovement.end())
                            , agentListWithPairs.begin()
                            , [] HD (const thrust::tuple<unsigned, unsigned>& values) {
                                // location first and agent index second
                                return thrust::make_pair<unsigned, unsigned>(thrust::get<1>(values), thrust::get<0>(values));
                            });
    }*/
        struct ZipComparator
{
    __host__ __device__
    inline bool operator() (const thrust::tuple<unsigned , unsigned, unsigned> &a, const thrust::tuple<unsigned, unsigned, unsigned> &b)
    {
        return thrust::get<2>(a) < thrust::get<2>(b);

    }
};
 //   const int NUM_OF_CITIES=1;
//    const int NUM_OF_ITERATIONS=1;
    PostMovement::PostMovement(unsigned NUM_OF_CITIES, unsigned NUM_OF_ITERATIONS,unsigned agents, double movedRatioInside, double movedRatioOutside, unsigned locations,unsigned print_on)
                    : generatorHelper(agents*0.8)
                    , offset1((locations/2) + 1)
                    , offset2((locations/2) + 1)
                    , movement1(static_cast<unsigned>(agents*0.8))
                    , movement2(static_cast<unsigned>(agents*0.8))
                    , flags(agents)
                    , scanResult(agents)
                    , copyOfPairs(agents)
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
        std::uniform_int_distribution<unsigned> randomLocation{0, (locations - 1)/2};
        auto getRandomLocation = [&]() -> unsigned { return randomLocation(generator); };
        std::uniform_real_distribution<decltype(movedRatioOutside)> randomUnit{0.0, 1.0}; 
        auto t01 = std::chrono::high_resolution_clock::now();
        for(unsigned i=0;i<NUM_OF_CITIES;i++){//reserving memory 
            locationAgentLists[i].reserve(agents*0.8);
            hostAgentLocations[i].reserve(agents*0.8);
            offsets[i].reserve((locations/2) + 1);
            placeToCopyAgentLengths[i].reserve(agents*0.8);
            hostMovements[i].reserve(agents*0.8);
            hostexChangeAgents[i].reserve(agents*0.8);
            exChangeAgents[i].reserve(agents*0.8);
            offsetForExChangeAgents[i].reserve(NUM_OF_CITIES+1);
            offsetForExChangeAgents[i].resize(NUM_OF_CITIES+1);
            exchangeHelperVectors[i].reserve(agents*0.8);
            stayedAngentsHelperVectors[i].reserve(agents*0.8);
            IncomingAgents[i].reserve(agents*0.8);
            hostIncomingAgents[i].reserve(agents*0.8);
            agentLocationAfterMovements[i].reserve(agents*0.8);
        }
        generatorHelper.reserve(agents*0.8);
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
        auto last_loop = 0;


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
             //   std::cout << "Generate locationagentlist \n ";

              //  thrust::copy(generatorHelper.begin(), generatorHelper.end(), std::ostream_iterator<unsigned>(std::cout, " "));
             //   std::cout << "\n";
                thrust::transform(generatorHelper.begin(), generatorHelper.end(), locationAgentLists[i].begin(), [i] (unsigned loc) {
                    return thrust::make_tuple(i, loc);
                });

                thrust::sequence(offset1.begin(), offset1.end());
                thrust::transform(offset1.begin(),offset1.end(), offset1.begin(), [&] (const auto& loc)  {
                    return (thrust::count(hostAgentLocations[i].begin(), hostAgentLocations[i].end(), loc));
                });
             //   thrust::copy(offset1.begin(), offset1.end(), std::ostream_iterator<unsigned>(std::cout, " "));

                thrust::exclusive_scan(offset1.begin(), offset1.end(), offset1.begin());
                    // offset1 = hostOffset1;
             //   std::cout << "Generated offset1\n ";
                thrust::copy(offset1.begin(), offset1.end(), offsets[i].begin());
             //   thrust::copy(offset1.begin(), offset1.end(), std::ostream_iterator<unsigned>(std::cout, " "));
             //   std::cout << "\n";
                offsets[i]=offset1;

            }
            auto t2 = std::chrono::high_resolution_clock::now();
            update_arrays_time += std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();


            
            // generate movement
            auto t3 = std::chrono::high_resolution_clock::now();
            for(unsigned i=0;i<NUM_OF_CITIES;i++){
                agentIDs[i].resize(vector_sizes[i]);
                exchangeHelperVectors[i].resize(vector_sizes[i]);
                stayedAngentsHelperVectors[i].resize(vector_sizes[i]);
                thrust::for_each(exchangeHelperVectors[i].begin(), exchangeHelperVectors[i].end(),[&] (auto &v) {
                    v=0;
                });
                thrust::for_each(stayedAngentsHelperVectors[i].begin(), stayedAngentsHelperVectors[i].end(),[&] (auto &v) {
                    v=0;
                });
            
                hostMovements[i].resize(vector_sizes[i]);
                thrust::transform(thrust::make_zip_iterator(agentIDs[i].begin(), hostAgentLocations[i].begin())
                                , thrust::make_zip_iterator(agentIDs[i].end(), hostAgentLocations[i].end())
                                , hostMovements[i].begin()
                                , [&](const auto& idLocPair)  {
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
                                        ,[&] (const auto &tup) {
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
                exChangeAgents[i].resize( movedAgentSizeFromCities[i]);
                hostexChangeAgents[i].resize( movedAgentSizeFromCities[i]);
                    thrust::for_each(thrust::make_zip_iterator(// this for each will put the leaving agents into the exChangeAgents[i].
                                hostMovements[i].begin(),
                                thrust::make_permutation_iterator(exChangeAgents[i].begin(), exchangeHelperVectors[i].begin())),
                        thrust::make_zip_iterator(
                                hostMovements[i].end(),
                                thrust::make_permutation_iterator(exChangeAgents[i].begin(), exchangeHelperVectors[i].end())),
                        [&] (const auto &tup) {
                            auto hostmovement =thrust::get<0>(tup);
                            auto city =thrust::get<2>(hostmovement);
                                if (city != i){
                                    auto id = thrust::get<0>(hostmovement);
                                    auto loc = thrust::get<1>(hostmovement);  
                                    thrust::get<1>(tup)= thrust::make_tuple<unsigned, unsigned,unsigned>(id,loc,city);
                                    thrust::get<0>(tup) = thrust::make_tuple(UINT_MAX ,UINT_MAX, UINT_MAX);
                                }
                        });
                    hostexChangeAgents[i] = exChangeAgents[i]; // to be able to print, this can be removed later

                    thrust::stable_sort(exChangeAgents[i].begin(),exChangeAgents[i].end(), ZipComparator());
                    thrust::host_vector<unsigned>cityIndex(exChangeAgents[i].size());
                    thrust::for_each(thrust::make_zip_iterator(
                                cityIndex.begin(),
                                exChangeAgents[i].begin())
                                ,thrust::make_zip_iterator(
                                cityIndex.end(),
                                exChangeAgents[i].end()),[&] (const auto &tup) {
                                    auto movement =thrust::get<1>(tup);
                                    thrust::get<0>(tup) = thrust::get<2>(movement);
                                    
                        });

                    thrust::sequence(offsetForExChangeAgents[i].begin(), offsetForExChangeAgents[i].end());
                    thrust::transform(offsetForExChangeAgents[i].begin(),offsetForExChangeAgents[i].end(),
                     offsetForExChangeAgents[i].begin(), [&] (const auto& loc)  {
                        return (thrust::count(cityIndex.begin(), cityIndex.end(), loc));
                    });
                    thrust::exclusive_scan(offsetForExChangeAgents[i].begin(),
                    offsetForExChangeAgents[i].end(), offsetForExChangeAgents[i].begin());
                    hostexChangeAgents[i] = exChangeAgents[i]; // to be able to print, this can be removed later
                    if(print_on){
                        std::cout << "After sorting: Agents moving from "<< i+1 << "st city :"<< movedAgentSizeFromCities[i] <<"\n ";
                        for(unsigned  j =0;j<  hostexChangeAgents[i].size();j++){
                            std::cout<<  " ID" <<thrust::get<0>(hostexChangeAgents[i][j]) << " city " << thrust::get<2>(hostexChangeAgents[i][j])<< " loc " << thrust::get<1>(hostexChangeAgents[i][j]);
                        }
                        std::cout << "\n";
                   }
            }
            auto t6 = std::chrono::high_resolution_clock::now(); 
            sorting_merging_arrays_after_movement += std::chrono::duration_cast<std::chrono::microseconds>(t6-t5).count();   
            auto t7 = std::chrono::high_resolution_clock::now(); 

          //  std::cout<<"\n"<< "\n";

     
            for(unsigned i=0;i<NUM_OF_CITIES;i++){//generate the locations after the movements
                
                unsigned incomingAgentsNumberToParticularCity=0;
                for (unsigned city =0;city<NUM_OF_CITIES;city++){
                    unsigned from =offsetForExChangeAgents[city][i];
                    unsigned to=offsetForExChangeAgents[city][i+1];
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
                        std::cout<<"from city "<<city<<"to city "<< i <<"start: "<<from <<" end : "<<to<<"\n";
                        thrust::for_each(thrust::make_zip_iterator(
                                        exChangeAgents[city].begin()+from,
                                        IncomingAgents[i].begin()+numberOfAgentsPutToIncomingAgents),
                                        thrust::make_zip_iterator(
                                        exChangeAgents[city].begin()+to,
                                        IncomingAgents[i].begin()+numberOfAgentsPutToIncomingAgents+to-from)
                                        ,[&] (const auto &tup) {                     
                                    thrust::get<1>(tup)=thrust::get<0>(tup);
                                    
                                });

                    numberOfAgentsPutToIncomingAgents+=to-from;

                
                }


 
                agentLocationAfterMovements[i].resize(vector_sizes[i]-movedAgentSizeFromCities[i]+incomingAgentsNumberToParticularCity);
                thrust::exclusive_scan(stayedAngentsHelperVectors[i].begin(), stayedAngentsHelperVectors[i].end(), stayedAngentsHelperVectors[i].begin());
                thrust::for_each(thrust::make_zip_iterator(//put the stayed agents into the agentLocationAfterMovements[i]
                                hostMovements[i].begin(),
                                thrust::make_permutation_iterator(agentLocationAfterMovements[i].begin(), stayedAngentsHelperVectors[i].begin())),
                                thrust::make_zip_iterator(
                                hostMovements[i].end(),
                                thrust::make_permutation_iterator(agentLocationAfterMovements[i].begin(), stayedAngentsHelperVectors[i].end())),
                        [&] (const auto &tup) {
                            auto hostmovement =thrust::get<0>(tup);
                            auto city =thrust::get<2>(hostmovement);
                            auto id = thrust::get<0>(hostmovement);
                            auto loc = thrust::get<1>(hostmovement);  
                            thrust::get<1>(tup)= thrust::make_tuple<unsigned, unsigned,unsigned>(id,loc,city);
            
                        });
                     
                thrust::for_each(thrust::make_zip_iterator(//this will put the incoming agents to the end of the agentLocationAfterMovements[i]
                    agentLocationAfterMovements[i].begin()+vector_sizes[i]-movedAgentSizeFromCities[i],
                    IncomingAgents[i].begin()),
                    thrust::make_zip_iterator(
                    agentLocationAfterMovements[i].end(),
                    IncomingAgents[i].end())
                    ,[&] (const auto &tup) {                     
                        thrust::get<0>(tup)=thrust::get<1>(tup);
              });


                //this is only needed for printing
                thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned>> hostagentLocationAfterMovement1(agentLocationAfterMovements[i].size());
                std::copy(agentLocationAfterMovements[i].begin(), agentLocationAfterMovements[i].end(), hostagentLocationAfterMovement1.begin());
                for(unsigned j=0;j<agentLocationAfterMovements[i].size()/* vector_sizes[i]-movedAgentSizeFromCities[i]+incomingAgentsNumberToParticularCity*/;j++){
                    auto id = thrust::get<0>(hostagentLocationAfterMovement1[j]);
                    auto loc = thrust::get<1>(hostagentLocationAfterMovement1[j]);
                    if(print_on)
                        std::cout << " ID " <<id << "  loc "<<  loc;  
                }
                if(print_on)
                    std::cout<<"\n";
                
                //prepare vectors for new loop
                //firstly the hostAgentLocation and the IDs:
                vector_sizes[i]=agentLocationAfterMovements[i].size();
                agentIDs[i].resize(vector_sizes[i]);//think about it, it is okay?
                hostAgentLocations[i].resize(vector_sizes[i]);
                thrust::for_each(thrust::make_zip_iterator(
                                agentIDs[i].begin(),
                                hostAgentLocations[i].begin(),
                                agentLocationAfterMovements[i].begin()),
                                thrust::make_zip_iterator(
                                agentIDs[i].end(),
                                hostAgentLocations[i].end()
                                ,agentLocationAfterMovements[i].end()),
                    [&] (const auto &tup) {
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
            last_loop += std::chrono::duration_cast<std::chrono::microseconds>(t8-t7).count();   
        }
       
        std::cout<<"update_arrays_time took "<<update_arrays_time<< " microseconds\n";
        std::cout<<"movement_time took "<<movement_time<< " microseconds\n";
        std::cout<<"sorting_merging_arrays_after_movement took "<<sorting_merging_arrays_after_movement/1000<< " microseconds\n";
        std::cout<<"last_loop took "<<last_loop<< " microseconds\n";


    }

    