#!/bin/bash -f
for iter in {2,4}
do  
    OMP_NUM_THREADS=32 mpirun -np 2 ~reguly/numawrap_omp2 ./mpi.exe 0 ${iter} >real_2_iter_${iter}.txt
    echo "real_2_iter_${iter}txt  done"
    OMP_NUM_THREADS=16 mpirun --oversubscribe -np 4 --bind-to numa ./mpi.exe 0 ${iter} >real_4_iter_${iter}.txt
    echo "real_4_iter_${iter}txt  done"

 

done
