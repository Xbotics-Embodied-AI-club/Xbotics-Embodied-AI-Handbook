"""把训练控制台日志里的 loss 画成曲线，作为微调收敛的证据图。

在讲12 2.3 节被引用。

lerobot-train 每 log_freq 步打印一行 `step:200 ... loss:0.206 ...`。
train_smolvla.sh 已把这份控制台输出 tee 到 scratch/train_smolvla.log，
这里正则抓出 (step, loss) 两列，画一条对数纵轴的曲线存进 result/。
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

here = Path(__file__).parent
log_path = here / "scratch" / "train_smolvla.log"
out_path = here / "result" / "loss_curve.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

# 每行形如 "step:200 smpl:13K ep:256 epch:16.00 loss:0.206 grdn:1.457 ..."，
# 抓 step 和 loss 两个数即可。
pattern = re.compile(r"step:(\d+)\s.*?loss:([0-9.]+)")
steps, losses = [], []
for line in log_path.read_text().splitlines():
    m = pattern.search(line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))

plt.figure(figsize=(7, 4))
plt.plot(steps, losses, color="#c0392b", linewidth=1.5)
plt.yscale("log")
plt.xlabel("训练步数 step")
plt.ylabel("flow-matching 损失 (log)")
plt.title("SmolVLA 全量微调 · SO-101 ReachCube")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, dpi=130)
print(f"最终 loss={losses[-1]:.4f} @ step {steps[-1]}  ->  {out_path}")
