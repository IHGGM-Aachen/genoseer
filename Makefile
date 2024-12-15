.PHONY: create_env install_dependencies

create_env:
	python3.12 -m venv venv; . venv/bin/activate; pip install -r requirements.txt

train_residue_autoencoder:
	 python './src/training_gae.py' --config_path './src/configs/training_gcn.yaml'

run_inference_residue_autoencoder:
	 python './src/inference_gae.py' --config_path './src/configs/training_gcn.yaml'

train_atomic_autoencoder:
	 python './src/training_gae.py' --config_path './src/configs/training_mpnn.yaml'

run_inference_atomic_autoencoder:
	 python './src/inference_gae.py' --config_path './src/configs/training_mpnn.yaml'