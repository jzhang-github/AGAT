for i in $(seq 0.3 0.1 0.9)
do
    RuRhIr_con=$(echo $i/3 | bc -l)
    Pd_max=$(echo 1-$i | bc -l)
    for j in $(seq 0.1 0.1 $Pd_max)
    do
        Pt_con=$(echo 1-$i-$j | bc -l)
        echo Ru${RuRhIr_con}Rh${RuRhIr_con}Ir${RuRhIr_con}Pd${j}Pt${Pt_con}
    done
done
