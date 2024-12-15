# Utilizing Protein Structure Graph Embeddings to Predict the Pathogenicity of Missense Variants

This repository houses the code for the research paper "Utilizing protein structure graph embeddings to predict the pathogenicity of missense variants", authored by Martin Danner, Matthias Begemann1, Miriam Elbracht, Ingo Kurth, and Jeremias Krause. With this code, users can transform pdb protein structures into datasets suitable for machine learning especially to train two autoencoder models: "MPNN Graph Autoencoder" and "GCN Autoencoder". These trained models facilitate the generation of embeddings applicable in downstream analysis, like the prediction of pathogenicity.

If you find this code helpful in your research, please consider citing our paper.

## Getting Started

### Prerequisites

- Python 3.12
- Python's virtual environment module

### Installation and Usage

In order to get up and running with this project, follow these steps:

1. Clone the repository in your local environment.

2. Navigate to the project directory using the terminal.

3. Execute the following command, which uses a [Makefile](./Makefile)
   to create a Python virtual environment (`venv`), activates the `venv`, and then installs necessary dependencies from the [requirements.txt](./requirements.txt) file:

   ```
   make create_env
   ```

4. You can manipulate the training and inference configurations as per your needs using the provided yaml files ([training_gcn.yaml](./src/configs/training_gcn.yaml), [training_mpnn.yaml](./src/configs/training_mpnn.yaml)).

5. To train the Residue Autoencoder, run:

   ```
   make train_residue_autoencoder
   ```

6. Run inference on the trained Residue Autoencoder by executing:

   ```
   make run_inference_residue_autoencoder
   ```

7. Similarly, to train and run inference on the Atomic Autoencoder, run the following commands respectively:

   ```
   make train_atomic_autoencoder
   ```

   ```
   make run_inference_atomic_autoencoder
   ```

### Citation

If you use the tool presented in this repository, please cite us:

```
Utilizing protein structure graph embeddings to predict the pathogenicity of missense variants 
Martin Danner, Matthias Begemann, Miriam Elbracht, Ingo Kurth, Jeremias Krause
```
