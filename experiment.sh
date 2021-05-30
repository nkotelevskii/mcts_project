#!/bin/bash
 
declare -a use_UCT=(false true)

for uct in ${use_UCT[@]}; do
    for i in {300..305}; do
        echo use_UCT_${uct}__seed_${i} 
        python run.py --save_name tree__use_UCT_${uct}__seed_${i}.pkl --seed ${i} --use_UCT ${uct}
        python evaluate.py --load_name tree__use_UCT_${uct}__seed_${i}.pkl --seed ${i}
    done
done


