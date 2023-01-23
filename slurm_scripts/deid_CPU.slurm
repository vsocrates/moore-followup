#!/bin/bash
#SBATCH --partition=gpu,day,week
#SBATCH --requeue
#SBATCH --job-name=incar_deid
#SBATCH --out="%x-%j.out"
#SBATCH --mem=40G
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --time=00-16:00:00
#SBATCH --mail-type=ALL


# do something
echo "I'm echoing to stdout"
echo "I'm echoing to stderr" 1>&2
echo "My JobID is ${SLURM_JOBID}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s)"

# module restore condacuda101
# module load CUDA/10.2.89

module load miniconda
conda activate deid_env
cd /home/vs428/Documents/Moore/ehr_deidentification/steps/forward_pass

python ./Forward_Pass_CPU.py /home/vs428/project/Incarceration_Data/notes/ed_notes_19_20_obi_deid_input_subset.jsonl /home/vs428/project/Incarceration_Data/ner_datasets/ed_notes_19_20_ner_subset.jsonl /home/vs428/project/Incarceration_Data/predictions/ed_notes_19_20_predictions_subset.jsonl /home/vs428/project/Moore_data/notes/ed_notes_19_20_obi_deid_input_subset_deid.jsonl /home/vs428/Documents/Moore/ehr_deidentification/steps/forward_pass/run/i2b2/predict_moore.json


# export PRODIGY_HOME="/home/vs428/Documents/Moore"

# python -m prodigy train cancer_exclusion_model --textcat moore_cancer_exclusion_GOLD  --label-stats --base-model en_core_web_trf --gpu-id 0 --verbose --training.max_steps 3000  # --config cancer_exclusion_config.cfg
# ~/Documents/n2c2_2022/annotations/n2c2_planSubsection_train_merged_v2.jsonl
# python -m prodigy train assessment_model --ner n2c2_assessment --label-stats --base-model en_core_web_trf --gpu-id 0
