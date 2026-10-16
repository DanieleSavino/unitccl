#!/bin/bash -l
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH -o logs/run.out
#SBATCH -e logs/run.err

module purge
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cmake/4.1.2
module load gcc/12.2

cd $SLURM_SUBMIT_DIR

export NCCL_HOME=$SLURM_SUBMIT_DIR/nccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
ldd ./build/unitccl_bench | grep nccl

source .venv/bin/activate

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_PROXY_DUMP_SIGNAL=10
#export NCCL_BINE_BUFFER_MANAGEMENT=SEND
#export NCCL_BINE_BUFFER_MANAGEMENT=DOUBLE_SEND
export NCCL_BINE_BUFFER_MANAGEMENT=SEND
export UNITCCL_CHECK=1

export UNITCCL_ITERS=40
export UNITCCL_WARMUP=10
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL,TUNING
export NCCL_DEBUG_FILE=$SLURM_SUBMIT_DIR/logs/nccl.%h.%p.log


#NCCL_PROTO=SIMPLE NCCL_ALGO=BINE mpirun -n 2 ./build/unitccl_bench Bcast 16
#NCCL_PROTO=SIMPLE NCCL_ALGO=BINE mpirun -n 4 ./build/unitccl_bench AllGather 256

# A
#CUDA_LAUNCH_BLOCKING=1 NCCL_PROTO=SIMPLE NCCL_ALGO=BINE mpirun -n 4 ./build/unitccl_bench AllGather 256
# B
#NCCL_P2P_READ_ENABLE=0 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 OMPI_MCA_pml=ob1 OMPI_MCA_osc=^ucx NCCL_PROTO=SIMPLE NCCL_ALGO=BINE mpirun -n 4 ./build/unitccl_bench AllGather 256

#OMPI_MCA_pml=ob1 OMPI_MCA_osc=^ucx \
	#CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
	#mpirun -n 4 compute-sanitizer --tool memcheck --target-processes all \
	  #--log-file logs/san.%q{OMPI_COMM_WORLD_RANK}.log ./build/unitccl_bench AllGather 1024

CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_FILE=logs/core.%h.%p \
	NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_PROTO=SIMPLE NCCL_ALGO=BINE \
	mpirun -n 2 ./build/unitccl_bench AllGather 16
