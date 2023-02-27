#!/bin/bash

#------------------------------------------------------------------------
#------------------------- setup ray cluster ----------------------------
#------------------------------------------------------------------------


cd $PBS_O_WORKDIR

UHOME=/home/594/ag9272
nodeDnsIps=`cat $PBS_NODEFILE | uniq`
headNodeDnsIp=`uname -n`
headNodeIp=`hostname -i`
rayDashboardPort=8888
rayPort=5711
rayPassword='5241590000000000'


cat > $PBS_O_WORKDIR/setupRayWorkerNode.sh << 'EOF'
#!/bin/bash -l
set -e
ulimit -s unlimited
cd $PBS_O_WORKDIR

headNodeIp=${1}
rayPort=${2}
rayPassword=${3}
UHOME=${4}
hostIpNPort=$headNodeIp:$rayPort

module purge
module load pbs
source $UHOME/.pyenv/versions/3.11.2/envs/ray/bin/activate

echo `uname -n`
echo `hostname -i`
echo `ray start --address=$hostIpNPort --redis-password=$rayPassword &`
echo Done.
EOF

chmod +x $PBS_O_WORKDIR/setupRayWorkerNode.sh

echo "Setting up Ray cluster......."
for nodeDnsIp in $nodeDnsIps
do
        if [[ $nodeDnsIp == $headNodeDnsIp ]]
        then
                echo -e "\nStarting ray cluster on head node..."
                module purge
                module load pbs
                source $UHOME/.pyenv/versions/3.11.2/envs/ray/bin/activate
                ray start --head --port=$rayPort
                sleep 5
        else
                echo -e "\nStarting ray cluster on worker node..."
                pbs_tmrsh ${nodeDnsIp} $PBS_O_WORKDIR/setupRayWorkerNode.sh $headNodeIp $rayPort $rayPassword $UHOME &
                sleep 5
        fi
done

rm $PBS_O_WORKDIR/setupRayWorkerNode.sh

echo -e "\nRay dashboard connection string:"
echo "ssh -L $rayDashboardPort:0.0.0.0:$rayDashboardPort $USER@gadi.nci.org.au"
