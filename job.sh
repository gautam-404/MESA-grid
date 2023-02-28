#!/bin/bash
 
#PBS -l ncpus=12480
#PBS -l mem=5200GB
#PBS -l jobfs=300GB
#PBS -q normal
#PBS -P ht06
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/ht06
#PBS -l wd
#PBS -M anuj.gautam@usq.edu.au
#PBS -m abe

source ~/.pyenv/versions/3.11.2/envs/ray/bin/activate
source ~/.bashrc

python raygrid.py
