#!/bin/bash

#------------------------------------------------------------------------
#------------------------- Ray Cluster Setup ----------------------------
#------------------------------------------------------------------------


cd $PBS_O_WORKDIR

UHOME=`eval echo "~$USER"`
nodeDnsIps=`cat $PBS_NODEFILE | uniq`
headNodeDnsIp=`echo $nodeDnsIps | awk '{print $1}'`
headNodeIp=`hostname -i`
rayDashboardPort=7711
rayPort=5711
headNodeIpNport="${headNodeIp}:${rayPort}"
redisPassword=$(uuidgen)


cat > $PBS_O_WORKDIR/setupRayWorkerNode.sh << 'EOF'
#!/bin/bash -l
set -e
ulimit -s unlimited
cd $PBS_O_WORKDIR

headNodeIpNport=${1}
redisPassword=${2}
UHOME=${3}

source $UHOME/.bashrc

thisNodeIp=`hostname -i`
ray start --address=$headNodeIpNport --num-cpus=28 --redis-password=$redisPassword --block &
EOF

chmod +x $PBS_O_WORKDIR/setupRayWorkerNode.sh

echo "Setting up Ray cluster......."
J=0
for nodeDnsIp in `echo $nodeDnsIps`
do
        if [[ $nodeDnsIp == $headNodeDnsIp ]]
        then
                echo -e "\nStarting ray cluster on head node..."
                source $UHOME/.bashrc
                ray start --head --num-cpus=28 --port=$rayPort \
                --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=${rayDashboardPort}
                sleep 3
        else
                echo -e "\nStarting ray cluster on worker node $J: $nodeDnsIp"
                pbs_tmrsh $nodeDnsIp $PBS_O_WORKDIR/setupRayWorkerNode.sh $headNodeIpNport $redisPassword $UHOME &
                echo Done.
                # sleep 1
        fi
        J=$((J+1))
done

echo "Ray cluster setup complete."
echo "Forward the dashboard port to localhost using the following command:"
echo "ssh -N -L 8880:0.0.0.0:$rayDashboardPort $USER@$headNodeDnsIp -J $USER@___.org.au"

sleep $J
rm $PBS_O_WORKDIR/setupRayWorkerNode.sh

