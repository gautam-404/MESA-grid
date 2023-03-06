#!/bin/bash
 
#PBS -N GYRE
#PBS -P 0201
#PBS -q default
#PBS -l select=4:ncpus=128:mem=512GB
#PBS -l walltime=24:00:00
#PBS -o $PBS_O_WORKDIR/job.out
#PBS -e $PBS_O_WORKDIR/job.err
#PBS -M anuj.gautam@usq.edu.au
#PBS -m abe

source ~/.bashrc
cd $PBS_O_WORKDIR

python raygrid.py
