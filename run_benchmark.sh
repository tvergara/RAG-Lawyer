#!/bin/bash

#SBATCH --job-name=python_benchmark
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --time=05:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

cd /home/tvergara/RAG-Lawyer
python -m benchmarks.ecthr-a

