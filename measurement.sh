#!/bin/bash -f

for j in {10000,20000,40000,80000,160000,320000,640000,1280000}
do 
    for i in {1,2,4,8}
    do       
        ./gpu.exe ${i} 10 ${j} 0 >> city${i}_agents${j}_iterations10_005_gpu.txt 
        echo "city${i}_agents${j}_iterations20_005.txt done"
          
    done
done