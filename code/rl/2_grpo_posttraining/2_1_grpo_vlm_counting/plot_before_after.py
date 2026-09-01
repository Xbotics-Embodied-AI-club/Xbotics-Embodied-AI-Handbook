"""把 GRPO 后训练前后的两项指标画成一张对照柱状图。

数据来自 result/2_1_grpo_vlm_counting/base_vs_adapter_summary.json
（SuperCLEVR 固定 200 题，训练前后各评一遍），这里只负责画图，不重跑评测。

原来这张图是有的，但出图脚本没跟着进仓。本文件按留存的 summary JSON 把它补回来，
同时接上全书统一字体。柱标按两位小数印，与讲义正文表格里的原始值（0.095）差在四舍五入，
讲15 的图注已经写明这一点，所以这里保持原样不动。
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 全书统一字体：西文 Times New Roman、中文宋体。字体路径走环境变量，取不到就报错停下。
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "assets" / "figures"))
import figstyle  # noqa: E402  —— 必须在 sys.path 补好之后再导入
FONT_NAME = figstyle.apply()

here = Path(__file__).parent
summary = json.loads(
    (here.parent / "result" / "2_1_grpo_vlm_counting" / "base_vs_adapter_summary.json").read_text())
out_path = here.parents[3] / "assets" / "figures" / "lecture15" / "ref" / "fig154-vlm-counting-before-after.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

METRICS = [("accuracy", "Counting accuracy"), ("format", "Answer format")]
BARS = [("base", "Base (Qwen2.5-VL-3B)", "#9AA7B1"), ("adapter", "After GRPO", "#2E7D32")]
WIDTH = 0.35

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for offset, (stage, label, color) in zip((-WIDTH / 2, WIDTH / 2), BARS):
    values = [summary[stage][key] for key, _ in METRICS]
    positions = [i + offset for i in range(len(METRICS))]
    ax.bar(positions, values, WIDTH, color=color, label=label)
    for x, v in zip(positions, values):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)

ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels([name for _, name in METRICS])
ax.set_ylabel(f"Rate on SuperCLEVR-{summary['sample_count']}")
ax.set_ylim(0, 1.0)
ax.set_title("GRPO post-training: VLM learns to count")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(out_path, dpi=150, metadata={"Font": FONT_NAME})
print(f"图已保存到 {out_path}")
