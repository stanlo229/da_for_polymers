#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --output=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/sklearn/SVM/Catalysis_Hein/slurm_batch_shuffled.out
#SBATCH --error=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/sklearn/SVM/Catalysis_Hein/slurm_batch_shuffled.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=24G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python opv_SVM_batch.py