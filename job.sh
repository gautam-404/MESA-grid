#!/bin/bash -l
 
#PBS -N MESA
#PBS -P 0201
#PBS -q default
#PBS -l select=4:ncpus=128:mem=512GB
#PBS -l walltime=12:00:00
#PBS -M anuj.gautam@usq.edu.au
#PBS -m abe

source ~/.bashrc
cd $PBS_O_WORKDIR

python raygrid.py
