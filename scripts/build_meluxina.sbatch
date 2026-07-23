#!/bin/bash -l
#SBATCH --account=p201236
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH -o logs/build.out
#SBATCH -e logs/build.err

module purge
module load env/release/2025.1
module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
module load CUDA/12.8
module load GCC/14.2.0
module load CMake/3.31.3-GCCcore-14.2.0
module load Python/3.13.1-GCCcore-14.2.0

set -e

export NCCL_HOME=/home/users/u104045/dev/unitccl/nccl
export NCCL_ROOT=/home/users/u104045/dev/unitccl/nccl

cd $SLURM_SUBMIT_DIR
python -m venv .venv
source .venv/bin/activate

cd $SLURM_SUBMIT_DIR/nccl
make -j src.build NVCC_GENCODE="-gencode arch=compute_80,code=sm_80"

#cd $SLURM_SUBMIT_DIR/vendor/fastest
#mkdir -p build
#cd build
#cmake ..
#make -j
#cd ../fastest_py
#pip install .

cd $SLURM_SUBMIT_DIR
make clean
make -j NVCC_GENCODE="-gencode arch=compute_80,code=sm_80"

#./scripts/build_wheel.sh
