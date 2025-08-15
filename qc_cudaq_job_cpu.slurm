#!/bin/bash
#SBATCH --job-name=quantum_ode_fast
#SBATCH --partition=compute                  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                  # More cores for faster execution
#SBATCH --mem=32G                           
#SBATCH --time=24:00:00                     # Reduced time with optimizations
#SBATCH --output=quantum_fast_%j.out        
#SBATCH --error=quantum_fast_%j.err         

# Activate Conda environment
source ~/.bashrc
conda activate cudaq-env

# Run the Python script
python qc_cloude.py
