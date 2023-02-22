#!/bin/bash
 
#PBS -l ncpus=8832
#PBS -l mem=16GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P a00
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/a00+scratch/a00
#PBS -l wd
  
module load python3
python3 grid.py $PBS_NCPUS > /g/data/a00/$USER/job_logs/$PBS_JOBID.log