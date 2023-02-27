#------------------------------------------------------------------------
#------------------------- setup ray cluster ----------------------------
#------------------------------------------------------------------------


cd $PBS_O_WORKDIR

nodeDnsIps=`cat $PBS_NODEFILE | uniq`
hostNodeDnsIp=`uname -n`
hostNodeIp=`hostname -i`
rayDashboardPort=8888
rayPort=6379
rayPassword='5241590000000000'


cat > $PBS_O_WORKDIR/setupRayWorkerNode.sh << 'EOF'
#!/bin/bash
set -e
ulimit -s unlimited
cd $PBS_O_WORKDIR
hostNodeIp=${1}
rayPort=${2}
rayPassword=${3}
hostIpNPort=$hostNodeIp:$rayPort
module purge
module load pbs
module load python3/3.11.0
module load openmpi/4.1.4
echo `which ray`
echo `uname -n`
echo `hostname -i`
echo `ray start --address=$hostIpNPort --num-cpus=$PBS_NCPUS --redis-password='5241590000000000'  --block &`
echo 
EOF

chmod +x $PBS_O_WORKDIR/setupRayWorkerNode.sh

echo "Setting up Ray cluster......."
for nodeDnsIp in ${nodeDnsIps}
do
        if [[ ${nodeDnsIp} == "${hostNodeDnsIp}" ]]
        then
                echo -e "\nStarting ray cluster on head node..."
                module purge
                module load pbs
                module load python3/3.11.0
                module load openmpi/4.1.4
                ray start --head --num-cpus=$PBS_NCPUS --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=${rayDashboardPort} --port=${rayPort}
                sleep 10
        else
                echo -e "\nStarting ray cluster on worker node..."
                pbs_tmrsh "${nodeDnsIp}" $PBS_O_WORKDIR/setupRayWorkerNode.sh "${hostNodeIp}" "${rayPort}" "${rayPassword}" &
                sleep 5
        fi
done



echo -e "\nCreating ray connection string ..."
echo "ssh -N -L ${rayDashboardPort}:${hostNodeDnsIp}:${rayDashboardPort} ${USER}@gadi.nci.org.au &" > ${PBS_O_WORKDIR}/connection_strings.txt
