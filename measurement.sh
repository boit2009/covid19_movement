#!/bin/bash -f

for j in {2560000,5120000,10240000,20480000}
do 
    for i in {1,2,4,8}
    do       
        ./gpu.exe ${i} 10 ${j} 0 >> city${i}_agents${j}_iterations10_005_gpu.txt 
        echo "city${i}_agents${j}_iterations20_005.txt done"
          
    done
done