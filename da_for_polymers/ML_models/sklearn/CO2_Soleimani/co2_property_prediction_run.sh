#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/sklearn/CO2_Soleimani/slurm.out
#SBATCH --error=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/sklearn/CO2_Soleimani/slurm.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python train.py --train_path ../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_train_[0-9].csv --validation_path ../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_valid_[0-9].csv --train_params_path co2_train_params.json --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_space.json --results_path ../../../training/CO2_Soleimani/BRICS --random_state 22