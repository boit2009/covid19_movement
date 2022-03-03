#!/bin/bash -f

for j in {10000,20000,40000,80000,160000}
do 
    for i in {1,2,3,4,5,6,7,8,9,10}
    do       
        ./a.out ${i} 10 ${j} 0 >> city${i}_agents${j}_iterations10_005.txt 
        echo "city${i}_agents${j}_iterations20_005.txt done"
          
    done
done