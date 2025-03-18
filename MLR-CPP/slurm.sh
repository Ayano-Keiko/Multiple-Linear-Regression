#!/usr/bin/bash
#SBATCH --partition=talon
#SBATCH --job-name=MLR
#SBATCH -o %j.txt

./main 10000 0.01
