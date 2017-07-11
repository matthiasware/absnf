#!/bin/bash
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:2
./performance_modulus
