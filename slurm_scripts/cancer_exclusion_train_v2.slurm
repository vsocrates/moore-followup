#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=moore_cancer_v2
#SBATCH --out="moorecancer_%x-%j.out"
#SBATCH --mem=35G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=00-06:00:00
#SBATCH --mail-type=ALL


# do something
echo "I'm echoing to stdout"
echo "I'm echoing to stderr" 1>&2
echo "My JobID is ${SLURM_JOBID}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s)"

# module restore condacuda101
module load CUDA/10.2.89
module load miniconda
# source activate prodigy
source activate prodigy_repl
cd /home/vs428/Documents/Moore 

export PRODIGY_HOME="/home/vs428/Documents/Moore"

# python -m prodigy train cancer_exclusion_model --textcat-multilabel moore_cancer_exclusion_GOLD  --label-stats --base-model en_core_web_trf --gpu-id 0 --verbose --training.max_steps 8000 --config config_textcat_multilabel.cfg

python -m spacy train cancer_prodigy_train_dataset/config_v2.cfg --paths.train cancer_prodigy_train_dataset/train.spacy --paths.dev cancer_prodigy_train_dataset/dev.spacy --gpu-id 0 --verbose --training.max_steps 12000 --output cancer_prodigy_train_dataset/
# ~/Documents/n2c2_2022/annotations/n2c2_planSubsection_train_merged_v2.jsonl
# python -m prodigy train assessment_model --ner n2c2_assessment --label-stats --base-model en_core_web_trf --gpu-id 0
