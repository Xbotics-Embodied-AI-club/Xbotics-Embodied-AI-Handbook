# G1 参考动作数据

本目录存放动作跟随实验（`1_3_g1_motion_tracking`）用的小体积参考动作资产。

课程使用的动作片段是 `marshal-arts`（一段武术视频），配套文件：

- `marshal-arts.mp4` — 原始人类动作视频。
- `marshal-arts.npz` — 重定向到宇树 G1 的参考动作（682 帧、50 FPS、29 个关节、30 个跟踪 body）。

生成管线（见 `../../1_2_video_to_g1_reference/`）：

1. GVHMR 从 `marshal-arts.mp4` 恢复人体动作；
2. GMR 把人体动作重定向到 Unitree G1；
3. `build_motion_npz.py` 打包写出 `marshal-arts.npz`。

体积较大的 GVHMR/GMR 中间产物不入库，落在：

```text
DATASETS_ROOT/models/trained/xbotics_rl_beyondmimic/reference_preprocess/marshal-arts/
```

训练 checkpoint 与 rollout 视频不属于本数据目录（分别见 `DATASETS_ROOT` 与 `../result/`）。
