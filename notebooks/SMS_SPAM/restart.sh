#!/bin/bash

#run(bash restart.sh) this file to remove any data files related to Cage and JL in data_pipeline, log, params folders
echo -n "remove any related data files(1 for cage, 2 for jl, 3 for both): "
read VAR

if [ $VAR == 1 ]
then 
rm data_pipeline/Cage/*
rm log/Cage/*
rm params/Cage/*
elif [ $VAR == 2 ]
then
rm data_pipeline/JL/*
rm log/JL/*
rm params/JL/*
else
rm data_pipeline/Cage/*
rm log/Cage/*
rm params/Cage/*
rm data_pipeline/JL/*
rm log/JL/*
rm params/JL/*
fi