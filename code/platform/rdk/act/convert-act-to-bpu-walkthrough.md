# 把训练好的 ACT 转成 BPU 模型(.hbm) —— 从头到尾对着敲

> 在 x86 训练机（开发机）上，把 LeRobot ACT checkpoint 导出 ONNX → 用地瓜 OE 工具链量化编译成 `.hbm`。
> 屏幕全程只出现 `~/...`（第 0 步软链接隐藏真实路径）。
> 例子是 **S600（nash-p）**；要出 **S100**，改 demo config 里 `type: nash-e`（命令行参数无效，见第 2 步）。

---

## 0. 隐藏真实路径（一次）

```bash
ln -sfn <EXPERIMENTS_ROOT> ~/exp
```

之后开发机命令都用 `~/exp/...`，不暴露真实目录。

---

## 1. 准备 OE 编译镜像（一次，27.5G）

> GPU 版 `ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`（27.5G）。tar 已放在 `~/Downloads/`，从零 load 即可。

先确认还没这个镜像（从零起点，应为空）：

```bash
sudo docker images | grep ai_toolchain
```

从本地 tar 导入（约几分钟）：

```bash
sudo docker load -i ~/Downloads/ai_toolchain_ubuntu_22_s100_s600_gpu_v3.7.0.tar
sudo docker images | grep ai_toolchain   # load 完出现，记下镜像名:标签，第 3 步要用
```

> tar 丢了要重下：`wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/ai_toolchain_ubuntu_22_s100_s600_gpu_v3.7.0.tar`
> 不想用 GPU：镜像名 `gpu`→`cpu`，第 3 步去掉 `--gpus all`。

---

## 2. 导出 ONNX + 量化校准数据（开发机，CPU）

```bash
cd ~/exp/lerobot/rdk/rdk_LeRobot_tools
HF_HUB_OFFLINE=1 ~/exp/.venv/bin/python export_bpu_actpolicy.py \
  --config ~/exp/lerobot/rdk/act/bpu_export_config_so101_cuboid_demo.yaml
```

- ⚠️ **只能给 `--config` 一个参数**：脚本在交给 draccus 前会把 `--export-path`/`--act-path`/`--type`/`--cal-num` 等命令行参数**全部删掉**（源码 `export_bpu_actpolicy.py` L120-143），所以这些命令行覆盖**完全无效**——一切设置只认 config YAML。
- demo config 里已设：`act_path`(原始 top/wrist checkpoint)、`export_path: $DATASETS_ROOT/models/bpu_export_demo`、`type: nash-p`(S600)、`cal_num: 100`、`onnx_sim: true`。要改 checkpoint / 输出路径 / **S100 换 `nash-e`** / 校准数，**改 config 文件**，别加命令行参数。
- 脚本会先 `rmtree` 清空 `export_path` 再写——demo 路径是专用空目录，安全（不会动到真实部署产物）。

跑完产出（在 `$DATASETS_ROOT/models/bpu_export_demo/`）：两个子模型各自的 `*.onnx` + 校准数据 + `build_*.sh`，外加 `build_all.sh` 和 `bpu_output/`（运行时归一化 npy）。

### 配置文件逐字段解析 `bpu_export_config_so101_cuboid_demo.yaml`

整条转换的"旋钮"全在这一个 YAML 里（命令行只认它）。按"输入 → 模型 → 量化 → 输出平台"四组看：

```yaml
dataset:                                    # ① 校准数据来源（拿真实样本去统计激活值的数值范围）
  repo_id: "so101/put_black_cuboid_into_basket"
  root: "$DATASETS_ROOT/hf-hub/lerobot/so101/put_black_cuboid_into_basket"
policy:                                     # ② 模型怎么读
  type: "act"                               #    策略类型 = ACT（决定用哪套导出逻辑）
  device: "cpu"                             #    导出在 CPU 上做（只需前向取图，不训练，不吃 GPU）
wandb:
  enable: false                             #    导出阶段不连 wandb
act_path: "$DATASETS_ROOT/models/so101_act_cuboid_topwrist/pretrained_model"   # ③ 训练好的 ACT checkpoint（被转的源模型）
export_path: "$DATASETS_ROOT/models/bpu_export_demo"                           # ④ 输出目录（脚本会先清空再写，所以用独立 demo 目录）
cal_num: 100                                # ⑤ 校准样本数：从数据集抽 100 帧喂模型，统计每层激活范围（越多越准、越慢；100 够用）
onnx_sim: true                              # ⑥ 导出后用 onnx-simplifier 折叠冗余算子，图更干净、利于上板编译
type: "nash-p"                              # ⑦ 目标 BPU 平台：S600=nash-p / S100=nash-e（决定 opset、量化粒度、.hbm 指令集）
combine_jobs: 6                             # ⑧ OE 编译器并行 job 数（多核加速编译，不影响结果）
```

**为什么是这些值**：

- **① dataset**：量化要把 FP32 压成 int16，得知道每层激活的真实数值分布——所以用**训练同分布的真实数据**做校准，不能凭空给范围。这里直接复用训练数据集。
- **③ act_path / ④ export_path**：源在哪、产物去哪。`export_path` 脚本**会先 `rmtree` 清空**，因此指一个专用空目录（`bpu_export_demo`），绝不指向已有模型目录。
- **⑤ cal_num=100**：校准是"统计"不是"训练"，100 帧足够覆盖激活分布；调大更稳但更慢。
- **⑦ type 是最关键开关**：`nash-p`(S600) 与 `nash-e`(S100) 对应**不同 BPU 架构**，编出的 `.hbm` 指令集不通用——**换板子只改这一行**。
- **⑥ onnx_sim / ⑧ combine_jobs**：纯工程项，simplify 让计算图更规整、combine_jobs 多核并行加速编译，都不改变模型数值。

> 一句话：这份 config = **「拿哪个 checkpoint（③）、用哪些真实数据校准（①⑤）、压给哪块 BPU（⑦）、产物放哪（④）」**，其余是工程旋钮。

---

## 3. 量化编译成 .hbm（OE Docker）

把上一步的导出目录挂进 OE 容器，跑 `build_all.sh`：

```bash
sudo docker run --rm --gpus all --network host --shm-size 15g \
  -v $DATASETS_ROOT/models/bpu_export_demo:/open_explorer -w /open_explorer \
  ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
  -c "bash build_all.sh"
```

- ⚠️ **镜像名就是 `ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`（`docker load` 出来的本地名，无 registry 前缀）**。写成 `registry.d-robotics.cc/deliver/...` 会被 docker 当远程仓库去 pull → 报 `401 Unauthorized`。以第 1 步 `docker images` 实际显示为准。
- 末尾是 `-c "bash build_all.sh"`（镜像 entrypoint 是 `/bin/bash`，不能直接写 `... 镜像 bash build_all.sh`）。
- CPU 版：去掉 `--gpus all`，镜像名 `gpu`→`cpu`。

---

## 4. 确认产物

```bash
ls $DATASETS_ROOT/models/bpu_export_demo/bpu_output/
```

应有：

```
BPU_ACTPolicy_VisionEncoder.hbm        # ResNet 吃图，输出 [1,512,15,20]
BPU_ACTPolicy_TransformerLayers.hbm    # 吃 states[1,6]+视觉特征，输出 Actions[1,100,6]
top_mean/std.npy  wrist_mean/std.npy  action_mean/std.npy(+unnormalize)  new_actions.npy
```

这套 `bpu_output/` 拷到板子就能用（部署见 `deploy_guide_so101_act.md` / `board_act_validate_benchmark_guide.md`）。

---

### 一页速查

```bash
# 0 隐藏路径（一次）
ln -sfn <EXPERIMENTS_ROOT> ~/exp

# 1 OE 镜像（没有才做）：tar 已在 ~/Downloads
sudo docker images | grep ai_toolchain || sudo docker load -i ~/Downloads/ai_toolchain_ubuntu_22_s100_s600_gpu_v3.7.0.tar

# 2 导出 ONNX（CPU；只 --config，其它命令行参数会被脚本丢弃）
cd ~/exp/lerobot/rdk/rdk_LeRobot_tools
HF_HUB_OFFLINE=1 ~/exp/.venv/bin/python export_bpu_actpolicy.py \
  --config ~/exp/lerobot/rdk/act/bpu_export_config_so101_cuboid_demo.yaml

# 3 编译 .hbm（OE Docker，GPU 版；镜像名无 registry 前缀，否则 401）
sudo docker run --rm --gpus all --network host --shm-size 15g \
  -v $DATASETS_ROOT/models/bpu_export_demo:/open_explorer -w /open_explorer \
  ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
  -c "bash build_all.sh"

# 4 看产物
ls $DATASETS_ROOT/models/bpu_export_demo/bpu_output/
```
