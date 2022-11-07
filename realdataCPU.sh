#!/bin/bash -f
for iter in {2,4,8,16,36}
do  
    OMP_NUM_THREADS=32 mpirun -np 2 ~reguly/numawrap_omp2 ./mpi.exe 0 ${iter} >real_2_iter_${iter}.txt
    echo "real_2_iter_${iter}txt  done"
    OMP_NUM_THREADS=16 mpirun --oversubscribe -np 4 --bind-to numa ./mpi.exe 0 ${iter} >real_4_iter_${iter}.txt
    echo "real_4_iter_${iter}txt  done"
    OMP_NUM_THREADS=8 mpirun --oversubscribe -np 8 --bind-to numa ./mpi.exe 0 ${iter} >real_8_iter_${iter}.txt.txt
    echo "real_8_iter_${iter}txt  done"
    OMP_NUM_THREADS=4 mpirun --oversubscribe -np 16 --bind-to numa ./mpi.exe 0 ${iter} >real_16_iter_${iter}.txt
    echo "real_16_iter_${iter}txt  done"
    OMP_NUM_THREADS=2 mpirun --oversubscribe -np 32 --bind-to numa ./mpi.exe 0 ${iter} >real_32_iter_${iter}.txt
    echo "real_32_iter_${iter}txt  done"
    OMP_NUM_THREADS=1 mpirun --oversubscribe -np 64 --bind-to numa ./mpi.exe 0 ${iter} >real_64_iter_${iter}.txt
    echo "real_64_iter_${iter}txt  done"
 

done
