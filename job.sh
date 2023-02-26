#!/bin/bash
 
#PBS -l ncpus=8832
#PBS -l mem=2200GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P ht06
#PBS -l walltime=00:10:00
#PBS -l storage=scratch/ht06
#PBS -l wd
#PBS -M anuj.gautam@usq.edu.au
#PBS -m abe
  
module load openmpi/4.1.4
source ~/.bashrc
mpiexec -n 184 --hostfile $PBS_NODEFILE ~/.pyenv/shims/python3 mpigrid.py $PBS_NCPUS > /scratch/ht06/$USER/workspace/MESA-grid/job_logs/$PBS_JOBID.log
