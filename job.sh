#!/bin/bash
 
#PBS -l ncpus=8832
#PBS -l mem=16GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P ht06
#PBS -l walltime=00:10:00
#PBS -l storage=scratch/ht06
#PBS -l wd
  
source ~/.bashrc
~/.pyenv/shims/python3 grid.py $PBS_NCPUS > /scratch/ht06/$USER/workspace/MESA-grid/job_logs/$PBS_JOBID.log
