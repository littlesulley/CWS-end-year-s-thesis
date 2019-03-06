#!/bin/bash

read -p "Please enter the following values: 'CUDA_VISIBLE_DEVICES', 'data_type' and 'batch_size' >> " CUDA data_type batch_size

if [ $data_type == "PKU" ]
then
    FILE="pku"
elif [ $data_type == "MS" ]
then
    FILE="msr"
elif [ $data_type == "City" ]
then 
    FILE="cityu"
else
    FILE="as"
fi

for dropout in 0.1 0.15 0.2 0.25
do
    for lr in 0.001 0.0015 0.002
    do
        for layers in 1 2
        do 
            for hidden_dim in 150 200
            do 
                for decay_rate in 0.9 0.95
                do
                    CUDA_VISIBLE_DEVICES=${CUDA} python main.py \
                    --train_path /data/nfsdata/hanqinghong/cws/training/${FILE}_train \
                    --dev_path /data/nfsdata/hanqinghong/cws/training/${FILE}_dev \
                    --eval_path /data/nfsdata/hanqinghong/cws/testing/${FILE}_test \
                    --shuffle \
                    --layers ${layers} \
                    --embed_dim 100 \
                    --hidden_dim ${hidden_dim} \
                    --dropout ${dropout} \
                    --batch_size ${batch_size} \
                    --lr ${lr} \
                    --epochs 100 \
                    --decay_rate ${decay_rate} \
                    --decay_step 5 \
                    --data_type ${data_type}
                    
                    sleep 1 &
                    wait 

                    CUDA_VISIBLE_DEVICES=${CUDA} python main.py \
                    --train_path /data/nfsdata/hanqinghong/cws/training/${FILE}_train \
                    --dev_path /data/nfsdata/hanqinghong/cws/training/${FILE}_dev \
                    --eval_path /data/nfsdata/hanqinghong/cws/testing/${FILE}_test \
                    --shuffle \
                    --layers ${layers} \
                    --embed_dim 100 \
                    --hidden_dim ${hidden_dim} \
                    --dropout ${dropout} \
                    --batch_size ${batch_size} \
                    --lr ${lr} \
                    --epochs 100 \
                    --decay_rate ${decay_rate} \
                    --decay_step 5 \
                    --data_type ${data_type} \
                    --model checkpoint/${data_type}/model

                    sleep 1 &
                    wait
                    
                    echo "==========Running script=========="
                    perl scripts/score gold/${FILE}_training_words gold/${FILE}_test_gold checkpoint/${data_type}/${data_type}_pred > checkpoint/${data_type}/score_${data_type}
                    echo "dropout: ${dropout}, lr: ${lr}, layer: ${layers}, hidden dim:${hidden_dim}, decay: ${decay_rate}. " $(tail -n 1 checkpoint/${data_type}/score_${data_type}) | tee -a checkpoint/${data_type}/${data_type}_log
                    
                    sleep 1 &
                    wait 

                done
            done
        done
    done
done




