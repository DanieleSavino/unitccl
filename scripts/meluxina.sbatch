#!/bin/bash -l
#SBATCH --account=p201236
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=00:00:30
#SBATCH -o logs/run.out
#SBATCH -e logs/run.err

module purge
module load env/release/2025.1
module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
module load CUDA/12.8
module load GCC/14.2.0
module load Python/3.13.1-GCCcore-14.2.0

cd $SLURM_SUBMIT_DIR

export NCCL_HOME=/home/users/u104045/dev/unitccl/nccl/build
export LD_LIBRARY_PATH=/home/users/u104045/dev/unitccl/nccl/build/lib:$LD_LIBRARY_PATH

source .venv/bin/activate

#NCCL_ALGO=RING mpirun -n 8 ./build/unitccl_bench Bcast 16
#NCCL_PROTO=SIMPLE NCCL_ALGO=BINE mpirun -n 8 ./build/unitccl_bench Bcast 16

#NCCL_ALGO=RING mpirun -n 8 ./build/unitccl_bench ReduceScatter 16
#NCCL_PROTO=SIMPLE NCCL_ALGO=RING mpirun -n 2 ./build/unitccl_bench Bcast 16
NCCL_PROTO=SIMPLE NCCL_ALGO=BINE mpirun -n 2 ./build/unitccl_bench AllGather 16
#NCCL_PROTO=SIMPLE NCCL_ALGO=BINE mpirun -n 2 ./build/unitccl_bench Bcast 16

#NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT NCCL_ALGO=BINE NCCL_PROTO=SIMPLE mpirun -n 2 ./build/unitccl_bench AllGather 64

#./scripts/run_tests.sh standalone

#./scripts/run_tests.sh scaling plot csv coll=Reduce proto=LL
