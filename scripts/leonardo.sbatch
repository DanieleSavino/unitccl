#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0724
#SBATCH -o logs/run.out
#SBATCH -e logs/run.err

module purge
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

cd $SLURM_SUBMIT_DIR

./scripts/run_tests.sh scaling plot coll=Bcast,Reduce,AllGather
