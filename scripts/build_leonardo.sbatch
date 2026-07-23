#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0724
#SBATCH -o logs/build.out
#SBATCH -e logs/build.err

module purge
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

cd $SLURM_SUBMIT_DIR
python -m venv .venv
source .venv/bin/activate

cd $SLURM_SUBMIT_DIR/nccl
make -j src.build NVCC_GENCODE="-gencode arch=compute_80,code=sm_80"

cd $SLURM_SUBMIT_DIR/vendor/fastest
mkdir -p build
cd build
cmake ..
make -j
cd ../fastest_py
pip install .

cd $SLURM_SUBMIT_DIR
make clean
make -j NVCC_GENCODE="-gencode arch=compute_80,code=sm_80"

./scripts/build_wheel.sh

