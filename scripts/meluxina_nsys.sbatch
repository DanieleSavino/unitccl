#!/bin/bash -l

#SBATCH --account=p201236
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH -o logs/run.out
#SBATCH -e logs/run.err

module purge
module load env/release/2025.1
module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
module load CUDA/12.8
module load GCC/14.2.0
module load Python/3.13.1-GCCcore-14.2.0

cd "$SLURM_SUBMIT_DIR"

export NCCL_HOME=/home/users/u104045/dev/unitccl/nccl/build
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"

source .venv/bin/activate

# NCCL settings
export NCCL_PROTO=SIMPLE

# Nsight settings
export NSYS_FLAGS="\
--trace=cuda,nvtx,osrt,mpi \
--cuda-memory-usage=true \
--sample=none \
--cpuctxsw=none \
--force-overwrite=true"

run_profile()
{
    ALGO=$1
    OP=$2
    SIZE=$3
    NAME="${OP,,}_${ALGO,,}_nsys"
    echo "Profiling $OP with $ALGO"
    export NCCL_ALGO=$ALGO
    mpirun -n 8 \
        nsys profile \
            $NSYS_FLAGS \
            -o "logs/${NAME}_rank%q{OMPI_COMM_WORLD_RANK}" \
            ./build/unitccl_bench \
            "$OP" \
            "$SIZE"
}


SIZE=16777216

run_profile BINE  Reduce $SIZE
run_profile RING  Reduce $SIZE

run_profile BINE  Bcast $SIZE
run_profile RING  Bcast $SIZE

#./scripts/run_tests.sh standalone

#./scripts/run_tests.sh scaling plot csv coll=Reduce proto=LL
