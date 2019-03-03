#!/bin/bash

DATA_TYPE=$1

if [ $DATA_TYPE == "MS" ]
then
    FILE="msr"
elif [ $DATA_TYPE == 'AS' ]
then
    FILE='as'
elif [ $DATA_TYPE == 'City' ]
then
    FILE='cityu'
else
    FILE='pku'
fi 

echo "Running script..."
perl scripts/score gold/${FILE}_training_words gold/${FILE}_test_gold checkpoint/${DATA_TYPE}/${DATA_TYPE}_pred > checkpoint/${DATA_TYPE}/score_${DATA_TYPE}

echo $(tail -n 1 checkpoint/${DATA_TYPE}/score_${DATA_TYPE})