# 人类视频 → G1 参考动作 工具链

动作跟随实验（`../1_3_g1_motion_tracking/`）要一段「G1 能跟」的参考动作。本工具链
把任意一段人类动作视频变成 G1 的关节参考轨迹，四步：

| 步骤 | 脚本 | 作用 |
|---|---|---|
| 1 | `video_to_human_motion.py` | 用 GVHMR 从单目视频恢复人体动作（SMPL-X） |
| 2 | `run_gvhmr_no_render.py` | GVHMR 推理的无渲染封装（服务器上跑不开窗口） |
| 3 | `human_motion_to_g1_reference.py` | 用 GMR 把人体动作重定向到宇树 G1（29 关节） |
| 4 | `build_motion_npz.py` | 打包成训练用的 `.npz`（关节轨迹 + body 位姿 + FPS） |

产物落组内共享数据目录：`../data/g1_reference_motions/<名字>.npz`。课程自带的
`marshal-arts.npz`（682 帧武术动作）就是用这条链生成的。

## 运行

```bash
cd code
uv sync --extra gpu_x86    # GVHMR / GMR 依赖随该 extra 安装
uv run python rl/1_rl_basics/1_0_video_to_g1_reference/video_to_human_motion.py
uv run python rl/1_rl_basics/1_0_video_to_g1_reference/human_motion_to_g1_reference.py
uv run python rl/1_rl_basics/1_0_video_to_g1_reference/build_motion_npz.py
```

体积大的中间产物（GVHMR/GMR 输出）不入库，落
`DATASETS_ROOT/models/trained/xbotics_rl_beyondmimic/reference_preprocess/` 下。
