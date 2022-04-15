#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/sklearn/RF/Swelling_Xu/slurm_batch.out
#SBATCH --error=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/sklearn/RF/Swelling_Xu/slurm_batch.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python swell_RF_cv_batch.py