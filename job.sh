#!/bin/bash
 
#PBS -l ncpus=4704
#PBS -l mem=500GB
#PBS -l jobfs=500GB
#PBS -q normal
#PBS -P ht06
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/ht06
#PBS -l wd
#PBS -M anuj.gautam@usq.edu.au
#PBS -m abe

export MESASDK_ROOT=/scratch/ht06/ag9272/workspace/software/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
export MESA_DIR=/scratch/ht06/ag9272/workspace/software/mesa-r22.11.1
export GYRE_DIR=$MESA_DIR/gyre/gyre
export PATH=$PATH:~/.local/bin

module restore MESA


python3 raygrid.py
