model_types=('RF' 'BRT' 'SVM')
for model in "${model_types}"
do
    # AUGMENTED SMILES
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_valid_[0-9].csv --feature_names Augmented_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_pair_BRICS --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_FP_radius_3_nbits_512 --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/fingerprint --random_state 22
    
    # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_manual --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_frag --random_state 22
    
    # AUGMENTED MANUAL FRAG
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_manual_aug --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_frag_aug --random_state 22
    
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/SMILES --random_state 22
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_SELFIES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/SELFIES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_BigSMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/BigSMILES --random_state 22
done
