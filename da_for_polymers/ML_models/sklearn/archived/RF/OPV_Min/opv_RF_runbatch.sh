#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/sklearn/RF/OPV_Min/slurm_batch.out
#SBATCH --error=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/sklearn/RF/OPV_Min/slurm_batch.err
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=24G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python opv_RF_batch.py