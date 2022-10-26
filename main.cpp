#include <cstdlib>
#include <sstream>
#include <fstream>
#include "setup.hpp"
#include <tuple>
#include <array>
#include <iostream>
#include "thrust/transform.h"
#include "thrust/scatter.h"
#include "thrust/set_operations.h"
#include "thrust/iterator/permutation_iterator.h"
#include "thrust/sequence.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <thrust/system/system_error.h>
#include <mpi.h>
#include <cmath>
#define NUM_OF_ITERATION 144

constexpr double agentLocRatio = 2.25;
void readInTxt(std::string filename, thrust::host_vector<unsigned> &vector_)
{
    std::string population_string;
    std::ifstream population_read(filename);
    getline(population_read, population_string);
    population_read.close();
    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;
    while ((pos = population_string.find(delimiter)) != std::string::npos)
    {
        token = population_string.substr(0, pos);
        // std::cout << token << std::endl;
        vector_.push_back(stoi(token));
        population_string.erase(0, pos + delimiter.length());
    }
}
void readInCSR(std::string filename, thrust::host_vector<unsigned> &csr_val, thrust::host_vector<unsigned> &csr_col, thrust::host_vector<unsigned> &csr_row)
{
    std::string bin, value_string, col_string, row_string, readin_helper;
    std::ifstream population_read(filename);
    getline(population_read, value_string);
    getline(population_read, bin);
    getline(population_read, col_string);
    getline(population_read, bin);
    getline(population_read, row_string);
    population_read.close();

    for (int i = 0; i < 3; i++)
    {
        if (i == 0)
            readin_helper = value_string;
        if (i == 1)
            readin_helper = col_string;
        if (i == 2)
            readin_helper = row_string;
        std::string delimiter = ",";
        size_t pos = 0;
        std::string token;
        while ((pos = readin_helper.find(delimiter)) != std::string::npos)
        {
            token = readin_helper.substr(0, pos);
            // std::cout << token << std::endl;
            if (i == 0)
                csr_val.push_back(stoi(token));
            if (i == 1)
                csr_col.push_back(stoi(token));
            if (i == 2)
                csr_row.push_back(stoi(token));
            readin_helper.erase(0, pos + delimiter.length());
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /* if ( size != 1 && size != 2 && size != 4 && size != 8) {
         std::cout << "Must run with 1,2,4 or 8 processes\n";
         MPI_Abort(MPI_COMM_WORLD,1);
     }*/
    std::string processnum = std::to_string(size);
    thrust::host_vector<thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> structForAgents;
    /*unique agent ID 0
    loc            1
    home partition ID 2
    destination partition ID 3
    iternumber amikor elmegy 4
    iternumber amikor hazajön 5*/
    unsigned urank = rank;
    unsigned NUM_OF_ITERATIONS = NUM_OF_ITERATION;
    srand(0);
    thrust::host_vector<unsigned> population, part_indexes, csr_val, csr_col, csr_row, IDs_helper;
    std::string population_filename = "CSR_modified_unsymmetric_population.txt";
    std::string CSR_file = "CSR_modified_unsymmetric.txt";
    std::string indexes_filename = "partition_indexes_" + processnum + ".txt";
    std::string ids_filename = "ids_start_" + processnum + ".txt";
    /*std::string population_filename = "csr_population_probe.txt";
    std::string CSR_file = "csr_probe.txt";
    std::string indexes_filename = "part_indexes_probe.txt";
    std::string ids_filename = "ids_start.txt";*/
    readInTxt(population_filename, population);
    //std::cout << rank << " poppulation size:" << population.size() << std::endl;
    readInTxt(indexes_filename, part_indexes);
    readInTxt(ids_filename, IDs_helper);
    readInCSR(CSR_file, csr_val, csr_col, csr_row);
   /* for (int i = 0; i < IDs_helper.size(); i++)
    {
        std::cout << IDs_helper[i] << std::endl;
    }*/
    std::cout << rank << " size:" << size << std::endl;
    unsigned agent_start_index, agent_end_index, NUM_OF_AGENTS;
    agent_start_index = IDs_helper[urank];
    agent_end_index = IDs_helper[urank + 1];
    NUM_OF_AGENTS = agent_end_index - agent_start_index;
    unsigned processed_number_of_agents = 0;
    unsigned number_of_counties_in_this_partition = 0;
    std::cout << rank << " NUM_OF_AGENTS:" << NUM_OF_AGENTS << std::endl;

    // init structForAgents
    thrust::device_vector<unsigned> agentID, fillWithRank, fillIterNumber/*, locs*/;
    agentID.reserve(NUM_OF_AGENTS * 2.0);
    agentID.resize(NUM_OF_AGENTS);
    fillWithRank.reserve(NUM_OF_AGENTS * 2.0);
    fillWithRank.resize(NUM_OF_AGENTS);
    fillIterNumber.reserve(NUM_OF_AGENTS * 2.0);
    fillIterNumber.resize(NUM_OF_AGENTS);
   // locs.reserve(NUM_OF_AGENTS * 2.0);
   // locs.resize(NUM_OF_AGENTS);
    structForAgents.reserve(NUM_OF_AGENTS * 2.0);
    thrust::for_each(structForAgents.begin(), structForAgents.end(),[rank] __host__ __device__ ( thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>& tup) {//fill the cityvalue in the beginning
            thrust::get<0>(tup)=rank;
            thrust::get<1>(tup)=rank; //to be inited
            thrust::get<2>(tup)=rank;  
            thrust::get<3>(tup)=rank;
            thrust::get<4>(tup)=rank; //to be inited
            thrust::get<5>(tup)=rank;  
    }); 
    structForAgents.resize(NUM_OF_AGENTS);
    
    thrust::sequence(agentID.begin(), agentID.end(), 0 + agent_start_index);
    thrust::fill(fillWithRank.begin(), fillWithRank.end(), urank);
    thrust::fill(fillIterNumber.begin(), fillIterNumber.end(), NUM_OF_ITERATIONS);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple( // filling the struct with initial values
                         agentID.begin(),
                         fillWithRank.begin(),
                         fillIterNumber.begin(),
                         structForAgents.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         agentID.end(),
                         fillWithRank.end(),
                         fillIterNumber.end(),
                         structForAgents.end())),
                     [] __host__ __device__(thrust::tuple<unsigned &, unsigned &, unsigned &, thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned> &> tup)
                     {   
                         unsigned loc = 0;
                         unsigned agentID_ = thrust::get<0>(tup);
                         unsigned fillWithRank_ = thrust::get<1>(tup);
                         unsigned fillIterNumber_ = thrust::get<2>(tup);
                         thrust::get<3>(tup) = thrust::make_tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>(agentID_, loc, fillWithRank_, fillWithRank_, fillIterNumber_, fillIterNumber_);
                     });

    for (unsigned row_ind = 0; row_ind < part_indexes.size(); row_ind++)
    {
        if (part_indexes[row_ind] == rank)
        { // if the process is dealing with these counties
            number_of_counties_in_this_partition += 1;
            for (unsigned col_ind = csr_row[row_ind]; col_ind < csr_row[row_ind + 1]; col_ind++)
            {
                unsigned number_of_agents_to_this_col_ind = csr_val[col_ind];
                unsigned dest_county = csr_col[col_ind];
                unsigned dest_partition = part_indexes[dest_county];
                if (dest_partition != rank){

                    for (unsigned i=0;i<number_of_agents_to_this_col_ind;i++){
                    unsigned leaving_part_time, arriving_part_time;
                    int r= (rand() % 15)+30;
                    int r2= (rand() % 15)+80;
                    //int r= (rand() % 2)+1;
                    //int r2= (rand() % 2)+5;
                    leaving_part_time = r;
                    arriving_part_time = r2;
                    thrust::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned> origAgent = structForAgents[processed_number_of_agents];
                    structForAgents[processed_number_of_agents]= thrust::make_tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>
                    (thrust::get<0>(origAgent), thrust::get<1>(origAgent), thrust::get<2>(origAgent), dest_partition, leaving_part_time, arriving_part_time);
                    processed_number_of_agents+=1;
                }
                }
                
                
            }
        }
    }
    for(unsigned  j =0;j<  structForAgents.size();j++){
       // if (j%10000==0){
        unsigned id =thrust::get<0>(structForAgents[j]);
        unsigned loc =thrust::get<1>(structForAgents[j]);
        unsigned home=thrust::get<2>(structForAgents[j]);
        unsigned dest =thrust::get<3>(structForAgents[j]);
        unsigned leave =thrust::get<4>(structForAgents[j]);
        unsigned arrive =thrust::get<5>(structForAgents[j]);
        
        //std::cout<<"rank"<< rank << " ID" <<id <<" loc"<<loc<< " home " << home<< " dest" <<dest <<" leave " << leave<<"arrive"<< arrive<<std::endl;
        //}
       
    }

     std::stringstream str;
     /*str << argv[1];
     unsigned NUM_OF_CITIES;
     str >> NUM_OF_CITIES;
     str.clear();
     str << argv[2];
     int NUM_OF_ITERATIONS2;
     str >> NUM_OF_ITERATIONS2;
     str.clear();
     str << argv[3];
     unsigned agents;
     str >> agents;*/
     str.clear();
     str << argv[1];
     unsigned print_on;
     str >> print_on;
     str.clear();
     str << argv[2];
     unsigned iter_exchange_number;
     str >> iter_exchange_number;
     str.clear();
     str << argv[3];
     unsigned outSideRatioDividedBy;
     str >> outSideRatioDividedBy;
     auto movedRatioInside=static_cast<double>(0.25);
     auto movedRatioOutside=static_cast<double>(0.05);
     movedRatioOutside = movedRatioOutside/ pow(2,outSideRatioDividedBy);
     auto locs= static_cast<unsigned>(static_cast<double>(NUM_OF_AGENTS) / agentLocRatio);
     if(rank == 0){
         std::cout << "used_parameters_NUM_OF_CITIES:" << size << "_NUM_OF_ITERATIONS:"<<NUM_OF_ITERATIONS <<
         "_movedRatioInside:"<< movedRatioInside << "_movedRatioOutside:"<< movedRatioOutside << "_locs:"<< locs << "_print_on:" << print_on <<  "iter_exchange_number:"<< iter_exchange_number<<"\n";
     }

    unsigned rank2 = rank;
    unsigned size2 = size;
    /*if (NUM_OF_CITIES != size)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }*/

    PostMovement p(size2, NUM_OF_ITERATIONS,  movedRatioInside, movedRatioOutside, locs, print_on, rank2, size2, iter_exchange_number,
     NUM_OF_AGENTS, structForAgents, agentID);
    //PostMovement p(NUM_OF_CITIES, NUM_OF_ITERATIONS, NUM_OF_AGENTS ,structForAgents, agentID, fillWithRank, fillIterNumber, countyID, print_on, iter_exchange_number);
}
