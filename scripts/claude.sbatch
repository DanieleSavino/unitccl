#!/bin/bash
#SBATCH --job-name=bine-ag
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=bine-ag-%j.out

# =============================================================================
# Bine vs Ring AllGather diagnostics on Leonardo (one 4-node allocation).
#
#   sbatch bine_ag_tests.sbatch
#   SECTIONS="main node4" sbatch bine_ag_tests.sbatch     # subset
#   NSYS=1 sbatch bine_ag_tests.sbatch                    # adds the nsys 1kB section
#
# Sections (default: all but nsys):
#   check   correctness of every algo (16 ranks) at odd/small/medium counts
#   main    16 ranks (4 nodes x 4 GPUs), 64 MiB/rank        <- the slow case
#   node4   4 ranks, ONE node, 64 MiB/rank (NVLink only: is the gap network-related?)
#   gpu2    2 ranks, one node, 64 MiB/rank, P2P and forced-NET transport
#   bufsize 16 ranks, NCCL_BUFFSIZE 1/4/16 MiB (slot/FIFO-limited?)
#   onegpu  4 ranks, one per node (flat layout, the paper's setting)
#   small   16 ranks, 1 kB / 16 kB / 256 kB per rank (fixed-overhead floor)
#   info    NCCL_DEBUG=INFO dumps (channels / tuning) for 16 ranks
#   nsys    (NSYS=1) nsys traces at 1 kB, 16 ranks
# =============================================================================

set -u

# ---- edit these -------------------------------------------------------------
BIN=${BIN:-/leonardo/home/userexternal/dsavino0/dev/unitccl/build/unitccl_bench}
NCCL_LIB=/leonardo/home/userexternal/dsavino0/dev/unitccl/nccl/build/lib/

# Applied to every run. Bine's LL/LL128 specializations only implement BUTTERFLY and the
# tuner picks LL for BINE, so the Simple protocol has to be forced (as in your earlier
# AllGather_SIMPLE runs). Ring gets the same protocol for a fair comparison.
COMMON_ENV=(NCCL_PROTO=SIMPLE)

# One entry per algorithm: "NAME|ENV ASSIGNMENTS".
# Variables are NCCL_BINE_VARIANT (TREE|BUTTERFLY|PAT) and
# NCCL_BINE_BUFFER_MANAGEMENT (BLOCK_BY_BLOCK|PERMUTATION|DOUBLE_SEND|SEND).
# TREE / PAT are easy to add; TREE tied the butterflies to within 0.3% earlier.
ALGOS=(
  "RING|NCCL_ALGO=RING"
    "BINE_SEND|NCCL_ALGO=BINE NCCL_BINE_VARIANT=BUTTERFLY NCCL_BINE_BUFFER_MANAGEMENT=SEND"
      "BINE_BBB|NCCL_ALGO=BINE NCCL_BINE_VARIANT=BUTTERFLY NCCL_BINE_BUFFER_MANAGEMENT=BLOCK_BY_BLOCK"
        # "BINE_TREE|NCCL_ALGO=BINE NCCL_BINE_VARIANT=TREE"
	  # "BINE_PAT_SEND|NCCL_ALGO=BINE NCCL_BINE_VARIANT=PAT NCCL_BINE_BUFFER_MANAGEMENT=SEND"
	  )
	  # -----------------------------------------------------------------------------

	  SECTIONS=${SECTIONS:-"check main node4 gpu2 bufsize onegpu small info"}
	  [[ "${NSYS:-0}" == 1 ]] && SECTIONS="$SECTIONS nsys"

	  ITERS=${ITERS:-40}
	  WARMUP=${WARMUP:-10}
	  TMO=${TMO:-300}                 # per-run timeout (s): experimental kernels may hang
	  CPUS=${SLURM_CPUS_PER_TASK:-8}

	  C64M=16777216                   # floats -> 64 MiB per rank
	  C1K=256                         # 1 kB
	  C16K=4096                       # 16 kB
	  C256K=65536                     # 256 kB

	  module purge
	  module load cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 gcc/12.2.0
	  export LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH"

	  OUT=${SLURM_SUBMIT_DIR:-.}/bine_results_${SLURM_JOB_ID:-local}
	  mkdir -p "$OUT/logs" "$OUT/nsys"
	  RES=$OUT/results.csv
	  echo "section,config,algo,ranks,ppn,count_floats,median_ns,min_ns,straggler_ns,GBps_recv_per_rank,status" > "$RES"

	  {
		    echo "job $SLURM_JOB_ID  nodes: $SLURM_JOB_NODELIST"
		      echo "bin: $BIN"; ls -l "$BIN"
		        echo "nccl lib: $NCCL_LIB"
			  nvidia-smi -L
			    nvidia-smi topo -m
		    } > "$OUT/env.txt" 2>&1

		    # nsys wrapper: one report per rank
		    cat > "$OUT/nsys_wrap.sh" <<'EOF'
#!/bin/bash
prefix=$1; shift
exec nsys profile -t cuda,nvtx,osrt --force-overwrite true \
     -o "${prefix}_rank${OMPI_COMM_WORLD_RANK}" "$@"
EOF
chmod +x "$OUT/nsys_wrap.sh"

want() { [[ " $SECTIONS " == *" $1 "* ]]; }

# run section config name "algo env" ranks ppn count [extra env ...]
run() {
	  local section=$1 config=$2 name=$3 spec=$4 n=$5 ppn=$6 count=$7; shift 7
	    local envs=(UNITCCL_ITERS=$ITERS UNITCCL_WARMUP=$WARMUP)
	      local spec_env; read -ra spec_env <<< "$spec"
	        envs+=("${COMMON_ENV[@]}" "${spec_env[@]}" "$@")   # later assignments win
		  local log="$OUT/logs/${section}_${config}_${name}.log"
		    local wrap=()
		      [[ "$section" == nsys ]] && wrap=("$OUT/nsys_wrap.sh" "$OUT/nsys/${config}_${name}")

		        echo "[$(date +%T)] $section $config $name  (n=$n ppn=$ppn count=$count)"
			  timeout -k 10 "$TMO" mpirun -n "$n" --map-by "ppr:${ppn}:node:PE=${CPUS}" --bind-to core \
				        -x LD_LIBRARY_PATH env "${envs[@]}" "${wrap[@]}" "$BIN" AllGather "$count" \
					      > "$log" 2>&1
			    local rc=$?

			      local st med mn sg bw
			        st=$(awk '/^status:/{$1=""; sub(/^ /,""); print}' "$log" | tr ',' ';')
				  med=$(awk '/^median:/{print $2}' "$log")
				    mn=$(awk '/^min:/{print $2}' "$log")
				      sg=$(awk '/^straggler:/{print $2}' "$log")
				        if [[ -z "$med" ]]; then
						    st="FAIL(rc=$rc)"; med=""; mn=""; sg=""; bw=""
						      else
							          bw=$(awk -v c="$count" -v n="$n" -v t="$med" 'BEGIN{printf "%.2f", c*4*(n-1)/t}')
								    fi
								      echo "$section,$config,$name,$n,$ppn,$count,$med,$mn,$sg,$bw,$st" >> "$RES"
							      }

							      # same test for every algorithm in ALGOS
							      sweep() {
								        local section=$1 config=$2 n=$3 ppn=$4 count=$5; shift 5
									  local entry name spec
									    for entry in "${ALGOS[@]}"; do
										        name=${entry%%|*}; spec=${entry#*|}
											    run "$section" "$config" "$name" "$spec" "$n" "$ppn" "$count" "$@"
											      done
										      }

										      # ------------------------------- sections ------------------------------------

										      if want check; then
											        for c in 31 4099 1048576; do
													    sweep check "n16_c${c}" 16 4 $c UNITCCL_CHECK=1 UNITCCL_ITERS=3 UNITCCL_WARMUP=1
													      done
												      fi

												      if want main; then
													        sweep main "n16_ppn4_c64M" 16 4 $C64M
													fi

													if want node4; then
														  sweep node4 "n4_ppn4_c64M" 4 4 $C64M
													  fi

													  if want gpu2; then
														    sweep gpu2 "n2_p2p_c64M" 2 2 $C64M
														      sweep gpu2 "n2_net_c64M" 2 2 $C64M NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
													      fi

													      if want bufsize; then
														        for b in 1048576 4194304 16777216; do
																    sweep bufsize "n16_buf$((b/1048576))M" 16 4 $C64M NCCL_BUFFSIZE=$b
																      done
															      fi

															      if want onegpu; then
																        sweep onegpu "n4_ppn1_c64M" 4 1 $C64M
																fi

																if want small; then
																	  for c in $C1K $C16K $C256K; do
																		      sweep small "n16_c${c}" 16 4 $c UNITCCL_ITERS=200 UNITCCL_WARMUP=50
																		        done
																		fi

																		if want info; then
																			  for c in $C1K $C64M; do
																				      sweep info "n16_c${c}" 16 4 $c NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV \
																					                UNITCCL_ITERS=3 UNITCCL_WARMUP=1
																				        done
																					  grep -hE "nChannels|Channel [0-9]+/|comm .* rank 0 nRanks|NCCL_(ALGO|PROTO|BUFFSIZE|BINE)|NCCL_BINE|defaulting" \
																						         "$OUT"/logs/info_*.log 2>/dev/null | sort -u > "$OUT/info_summary.txt"
																				  fi

																				  if want nsys; then
																					    command -v nsys >/dev/null || { echo "nsys not found in PATH" >&2; }
																					      sweep nsys "n16_c${C1K}" 16 4 $C1K UNITCCL_ITERS=20 UNITCCL_WARMUP=5
																					        for f in "$OUT"/nsys/*.nsys-rep; do
																							    [[ -e "$f" ]] && nsys export --type sqlite --force-overwrite true -o "${f%.nsys-rep}.sqlite" "$f" >/dev/null 2>&1
																							      done
																						      fi

																						      # ------------------------------- summary -------------------------------------
																						      {
																							        echo
																								  echo "===== summary (ratio = median / RING median in the same config) ====="
																								    awk -F, '
																								        NR==FNR { if (FNR>1 && $3=="RING" && $7!="") r[$1","$2]=$7; next }
																									    FNR>1 {
																									          k=$1","$2
																										        if ($7=="") { printf "%-8s %-20s %-12s  %s\n",$1,$2,$3,$11; next }
																												      ratio = (k in r) ? sprintf("x%.2f", $7/r[k]) : "-"
																												            printf "%-8s %-20s %-12s %10.1f us  %7s GB/s  %7s  %s\n",$1,$2,$3,$7/1000,$10,ratio,$11
																													        }' "$RES" "$RES"
																													} | tee "$OUT/summary.txt"

																													echo
																													echo "results: $RES"
																													echo "logs:    $OUT/logs"

