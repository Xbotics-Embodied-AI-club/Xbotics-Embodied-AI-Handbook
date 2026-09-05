#!/usr/bin/env bash
# 训练结束后，对最终 checkpoint 在**三个仿真场景各跑一次足量评测**，出要写进讲稿的成功率。
#
# 为什么是三个场景而不是一个：训练集里仿真侧有三个任务（cube40 / cube20 / cylinder40），
# 只评 cube40 得到的数字**不代表这份策略**。用户的验收线是"仿真里 ≥70%"，
# 那就得三个都量，并且分别报 —— 合成一个平均数会把某一项的失败藏起来。
#
# 为什么不受"评测耗时 ≤ 训练 1/10"的约束：那条管的是训练途中的点，用来画趋势；
# 这一步是出结论的，50 局给出 2% 的分辨率（二项 95% 区间 p=0.5 时约 ±14 个百分点）。
#
# ★ 四个 `--env.*` 必须与训练时逐字一致，否则量的是另一套设置下的成功率。
#   含义与"不给会怎样"见 train_smolvla.sh 末尾。
# ★ `--env.task_description` 不给 ⇒ 仿真器按场景查它自己那张表，取值与数据集
#   tasks.parquet 逐字相同。**不要在这里手写指令**：手写与数据集差一个字，
#   量到的就是"策略没见过这句话"时的成功率。
#
# 用法：bash eval_final_3scene.sh <checkpoint 目录> [<每场景局数>]
set -euo pipefail

CKPT="${1:?用法: bash eval_final_3scene.sh <checkpoint 目录> [<局数>]}"
N="${2:-50}"
# 指定 VENV 就用它 bin 下的命令，不指定就用 PATH 上的（`uv sync` 之后即在 PATH）。
BIN="${VENV:+$VENV/bin/}"
OUT="${OUT_DIR:-${BASE:-$DATASETS_ROOT/so101-sft}/outputs/eval-3scene}"

export CUDA_VISIBLE_DEVICES="${GPUS:-2}"
# 代理不在这里写死：`http_proxy` 由机器的 /etc/environment 预置（AGENTS.md）。
# ⚠️ 非登录 shell 读不到 /etc/environment —— 用 nohup/setsid 后台起时若拉不到 HF，
#    在**调用这个脚本的那一侧**导出，不要把某台机器的代理地址钉进脚本。

[ -f "$CKPT/config.json" ] || { echo "★ $CKPT 不像是 pretrained_model 目录"; exit 1; }

for ENV_ID in SO101PickPlaceCube40-v1 SO101PickPlaceCube20-v1 SO101PickPlaceCylinder40-v1; do
  echo "===== $ENV_ID（$N 局）====="
  "${BIN}lerobot-eval" \
    --policy.path="$CKPT" \
    --policy.device=cuda \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --env.type=so101_sim \
    --env.task="$ENV_ID" \
    --env.control_mode=pd_joint_pos \
    --env.observation_width=640 \
    --env.observation_height=480 \
    --env.episode_length=500 \
    --env.fps=30 \
    --eval.n_episodes="$N" \
    --eval.batch_size=5 \
    --output_dir="$OUT/$ENV_ID" 2>&1 | tail -3
done

echo
echo "===== 三场景汇总（每场景 $N 局）====="
"${BIN}python" - "$OUT" "$N" <<'PY'
import json
import sys
from pathlib import Path
out, n = Path(sys.argv[1]), int(sys.argv[2])
rows = []
for env_id in ("SO101PickPlaceCube40-v1", "SO101PickPlaceCube20-v1", "SO101PickPlaceCylinder40-v1"):
    f = out / env_id / "eval_info.json"
    if not f.is_file():
        print(f"  {env_id}: 没有 eval_info.json —— 这一场没跑完"); continue
    doc = json.loads(f.read_text())
    # 顶层键是 `overall`。取错键时 `.get` 给 None、脚本照常打印就过去了 ——
    # 那正是「空结果被当成结论」，所以取不到直接报错。
    if "overall" not in doc:
        sys.exit(f"★ {f} 里没有 `overall`，实际顶层键是 {list(doc)}")
    agg = doc["overall"]
    rows.append((env_id, agg["pc_success"], agg["avg_max_reward"]))
    print(f"  {env_id:30s} pc_success={agg['pc_success']:5.1f}%  avg_max_reward={agg['avg_max_reward']:.3f}")
if len(rows) == 3:
    lo = min(r[1] for r in rows)
    avg = sum(r[1] for r in rows) / 3
    print(f"\n  三场景平均 {avg:.1f}%，最低那一项 {lo:.1f}%")
    # 验收线按**最低的那一项**判，不按平均：平均会把某一项的失败藏起来。
    print("  ⇒ " + ("三项都 ≥70%，达线" if lo >= 70 else
                    f"未达线 —— 最低的一项只有 {lo:.1f}%（验收按最低项判，不按平均）"))
print("EVAL_FINAL_3SCENE_END")
PY
