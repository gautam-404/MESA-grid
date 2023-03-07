#!/bin/bash -l
 
#PBS -N GYREmedium
#PBS -P 0201
#PBS -q default
#PBS -l select=1:ncpus=128:mem=128GB
#PBS -l walltime=72:00:00
#PBS -M anuj.gautam@usq.edu.au
#PBS -m abe

source ~/.bashrc
cd $PBS_O_WORKDIR

python raygrid.py
