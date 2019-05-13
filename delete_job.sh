for i in `seq 0 1`
do
    for j in  `seq 0 4`
    do
        kubectl delete jobs job-${i}-${1}-${j}
    done
done
