#!/usr/bin/env bash
# run_eval_queue.sh
# 每张 GPU 一次仅跑一个 eval_alignscore 任务；自动排队全部组合。

set -u  # 如需严格出错退出可加：set -euo pipefail

########################
# 配置区
########################
kv_size=(64 128 256 512 1024 2048)
methods=(SnapKV_1-0 pyramidkv_1-0 SnapKV_128-64 SnapKV_256-128)
datasets=(multi_news qmsum gov_report)
gpus=(0 1 2 3 4 5 6 7)   # 可用 GPU ID；按需改
log_dir="logs_eval"
mkdir -p "$log_dir"

########################
# 生成任务列表
########################
tasks=()
for kv in "${kv_size[@]}"; do
  for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
      tasks+=("$kv|$dataset|$method")
    done
  done
done

########################
# GPU -> PID 映射
########################
declare -A GPU2PID   # 当前运行在该 GPU 上的后台任务 PID；空表示空闲

########################
# 启动一个任务（异步）
# 参数：gpu kv dataset method
########################
launch_task () {
  local gpu="$1" kv="$2" dataset="$3" method="$4"
  local log_file="$log_dir/${dataset}_${method}_kv${kv}.log"

  echo "$(date '+%F %T') [LAUNCH] gpu=$gpu kv=$kv dataset=$dataset method=$method -> $log_file"

  (
    python3 eval_alignscore.py --gpu "$gpu" --dataset "$dataset" --method "$method" --kv_size "$kv"
  ) >"$log_file" 2>&1 &

  local pid=$!
  GPU2PID["$gpu"]=$pid
}

########################
# 清理已完成任务，释放 GPU
########################
reap_finished () {
  local g pid
  for g in "${!GPU2PID[@]}"; do
    pid="${GPU2PID[$g]}"
    # 若 PID 不存在(任务完成)，kill -0 会失败
    if ! kill -0 "$pid" 2>/dev/null; then
      # wait 一下拿到退出码（忽略错误输出）
      wait "$pid" 2>/dev/null
      unset GPU2PID["$g"]
      echo "$(date '+%F %T') [FREE] gpu=$g 任务完成。"
    fi
  done
}

########################
# 主调度循环
########################
for t in "${tasks[@]}"; do
  IFS='|' read -r kv dataset method <<<"$t"

  # 找空闲 GPU；如无空闲，轮询等待
  while :; do
    reap_finished

    # 寻找空闲 GPU
    free_gpu=""
    for g in "${gpus[@]}"; do
      if [[ -z "${GPU2PID[$g]+x}" ]]; then
        free_gpu="$g"
        break
      fi
    done

    if [[ -n "$free_gpu" ]]; then
      launch_task "$free_gpu" "$kv" "$dataset" "$method"
      break  # 派发成功，处理下一个任务
    fi

    sleep 1
  done
done

########################
# 等待所有剩余任务完成
########################
echo "$(date '+%F %T') 等待剩余任务..."
for pid in "${GPU2PID[@]}"; do
  wait "$pid"
done

echo "✅ 全部任务完成；日志目录：$log_dir"
