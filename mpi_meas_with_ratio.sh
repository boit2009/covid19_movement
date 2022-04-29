#!/bin/bash -f
for outsideratio in {1,3,4,5,6,7,8,9,10}
    do
    for iter in {1,2,4,8,16}
    do
        for j in {100000,200000}
        do 
            for i in {1,2}
            do       
                mpirun -np ${i} ~/numawrap ./mpi.exe ${i} 16 ${j} 0 ${iter} ${outsideratio}>> city${i}_agents${j}_iterations16_${outsideratio}_${iter}_MPI.txt 
                echo "city${i}_agents${j}_iterations16_${outsideratio}_${iter}_MPI.txt  done"
                
            done
        done
    done
done