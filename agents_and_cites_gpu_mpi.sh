#!/bin/bash -f

for agent_num in {1000000,2000000}
do 
    for process_city_num in {1,2}
    do       
         mpirun --oversubscribe -np ${process_city_num} ./numawrap ./gpu.exe ${process_city_num} 16 ${agent_num} 0 1 0>> city${process_city_num}_agents${agent_num}_iter_per_communication_outsideratio_MPI.txt 
                echo "city${process_city_num}_agents${agent_num}_iterations16_outsideratio_005_dividedby2pow_iter_per_communication_MPI.txt  done"
        
    done
done