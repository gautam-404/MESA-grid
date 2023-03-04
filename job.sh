#!/bin/bash
 
#PBS -l ncpus=6240
#PBS -l mem=2TB
#PBS -l jobfs=300GB
#PBS -q normal
#PBS -P ht06
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/ht06
#PBS -l wd
#PBS -M anuj.gautam@usq.edu.au
#PBS -m abe

source ~/.bashrc

python raygrid.py
