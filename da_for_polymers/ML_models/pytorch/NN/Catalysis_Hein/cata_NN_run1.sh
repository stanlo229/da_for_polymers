#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --output=/project/6025683/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/NN/Catalysis_Hein/slurm.out
#SBATCH --error=/project/6025683/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/NN/Catalysis_Hein/slurm.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python cata_NN.py --n_hidden 256 --n_embedding 128 --drop_prob 0.3 --learning_rate 1e-3 --train_batch_size 128