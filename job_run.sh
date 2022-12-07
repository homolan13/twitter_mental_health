#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=yanis.schaerer@students.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=FAIL,END

# Job name 
#SBATCH --job-name="BERT for NLP"

# Runtime and memory
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=2G

#SBATCH --cpus-per-task=8

# Partition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:2

# Install dependencies #
singularity exec --nv docker://pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime pip install -U scikit-learn
singularity exec --nv docker://pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime pip install -U transformers

# Run script #
singularity exec --nv docker://pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime python bert.py