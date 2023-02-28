#!/bin/bash

#------------------------------------------------------------------------
#------------------------- setup ray cluster ----------------------------
#------------------------------------------------------------------------


cd $PBS_O_WORKDIR

UHOME=/home/594/ag9272
nodeDnsIps=`cat $PBS_NODEFILE | uniq`
headNodeDnsIp=`uname -n`
headNodeIp=`hostname -i`
# rayDashboardPort=7711
rayPort=5711
headNodeIpNport="${headNodeIp}:${rayPort}"
redisPassword=$(uuidgen)
# export redisPassword=$redisPassword


cat > $PBS_O_WORKDIR/setupRayWorkerNode.sh << 'EOF'
#!/bin/bash -l
set -e
ulimit -s unlimited
cd $PBS_O_WORKDIR

############ MESA environment variables ###############
export MESASDK_ROOT=/scratch/ht06/ag9272/workspace/software/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
export MESA_DIR=/scratch/ht06/ag9272/workspace/software/mesa-r22.11.1
export OMP_NUM_THREADS=16      
export GYRE_DIR=$MESA_DIR/gyre/gyre
#######################################################

headNodeIpNport=${1}
redisPassword=${2}
UHOME=${3}

source $UHOME/.pyenv/versions/3.11.2/envs/ray/bin/activate

thisNodeIp=`hostname -i`
echo `ray start --address=$headNodeIpNport --num-cpus=48 --redis-password=$redisPassword --block`
EOF

chmod +x $PBS_O_WORKDIR/setupRayWorkerNode.sh

echo "Setting up Ray cluster......."
J=0
for nodeDnsIp in `echo ${nodeDnsIps}`
do
        if [[ ${nodeDnsIp} == "${headNodeDnsIp}" ]]
        then
                echo -e "\nStarting ray cluster on head node..."
                source $UHOME/.bashrc
                source $UHOME/.pyenv/versions/3.11.2/envs/ray/bin/activate
                ray start --head --num-cpus=48 --port=$rayPort
                sleep 3
        else
                echo -e "\nStarting ray cluster on worker node $J"
                pbs_tmrsh ${nodeDnsIp} $PBS_O_WORKDIR/setupRayWorkerNode.sh ${headNodeIpNport} $redisPassword $UHOME &
                echo Done.
                sleep 1
        fi
        J=$((J+1))
done

sleep 10
rm $PBS_O_WORKDIR/setupRayWorkerNode.sh
