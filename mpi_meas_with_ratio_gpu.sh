#!/bin/bash -f
for outsideratio in {0,2}
    do
    for iter_per_communication in {1,2,4,8,16}
    do
        for agent_num in {100000,1000000,10000000,100000000}
        do 
            for process_city_num in {1,2}
            do       
                mpirun --oversubscribe -np ${process_city_num} ./numawrap ./gpu.exe ${process_city_num} 16 ${agent_num} 0 ${iter_per_communication} ${outsideratio}>> city${process_city_num}_agents${agent_num}_iter_per_communication${iter_per_communication}_outsideratio${outsideratio}_MPI.txt 
                echo "city${process_city_num}_agents${agent_num}_iterations16_outsideratio_005_dividedby2pow${outsideratio}_iter_per_communication${iter_per_communication}_MPI.txt  done"
                
            done
        done
    done
done