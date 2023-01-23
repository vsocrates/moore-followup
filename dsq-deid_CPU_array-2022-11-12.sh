#!/bin/bash
#SBATCH --output dsq-deid_CPU_array-%A_%2a-%N.out
#SBATCH --array 0-13
#SBATCH --job-name dsq-deid_CPU_array
#SBATCH --partition=gpu,day,week --mem=50G --gpus=0 --nodes=1 --time=00-13:00:00 --mail-type=ALL


export LD_LIBRARY_PATH=/gpfs/milgram/project/rtaylor/vs428/conda_envs/deid_env/lib:$LD_LIBRARY_PATH

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/home/vs428/Documents/Moore/deid_CPU_array.txt --status-dir /gpfs/milgram/home/vs428/Documents/Moore

