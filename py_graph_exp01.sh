#! /bin/bash
#$ -t 1-5
source activate /cbica/home/shenr/envs/graph
cd /cbica/home/shenr/code
let FOLD=${SGE_TASK_ID}-1
echo processing $FOLD
python3 MCMC_train_exp01.py -f $FOLD
echo py_graph_exp01.sh Finished!