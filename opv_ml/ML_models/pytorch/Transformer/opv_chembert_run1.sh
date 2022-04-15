#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --output=/project/6025683/stanlo/opv_project/opv_ml/ML_models/pytorch/Transformer/slurm.out
#SBATCH --error=/project/6025683/stanlo/opv_project/opv_ml/ML_models/pytorch/Transformer/slurm.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=180G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python opv_chembert.py --drop_prob 0.3 --learning_rate 1e-2 --train_batch_size 128