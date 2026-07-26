from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_module
import hydra.utils as hydra_utils
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
import hmr4d.model.gvhmr.pipeline.gvhmr_pipeline
import hmr4d.model.gvhmr.utils.endecoder
import hmr4d.network.gvhmr.relative_transformer
from hmr4d.utils.geo.hmr_cam import convert_K_to_K4, create_camera_sensor, estimate_K, get_bbx_xys_from_xyxy
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.net_utils import detach_to_cpu
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
from hmr4d.utils.preproc.slam import SLAMModel
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer

CRF = 23


def _resolve_checkpoint_root(checkpoint_root: str | Path | None) -> Path:
    if checkpoint_root is None:
        checkpoint_root = (
            Path(os.environ["DATASETS_ROOT"]) / "models" / "downloaded" / "gvhmr" / "inputs" / "checkpoints"
        )
    expected = Path(checkpoint_root).resolve()

    required = [
        Path("body_models/smpl/SMPL_NEUTRAL.pkl"),
        Path("body_models/smplx/SMPLX_NEUTRAL.npz"),
        Path("gvhmr/gvhmr_siga24_release.ckpt"),
        Path("hmr2/epoch=10-step=25000.ckpt"),
        Path("vitpose/vitpose-h-multi-coco.pth"),
        Path("yolo/yolov8x.pt"),
    ]
    if all((expected / path).is_file() for path in required):
        return expected

    missing = [str(path) for path in required if not (expected / path).is_file()]
    raise RuntimeError(
        "GVHMR checkpoint bundle is incomplete. The classroom runner expects the upstream "
        f"layout at {expected}. Missing: "
        + ", ".join(missing)
    )


def _copy_video_to_gvhmr_workspace(video_path: Path, cfg) -> None:
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if Path(cfg.video_path).exists() and get_video_lwh(video_path)[0] == get_video_lwh(cfg.video_path)[0]:
        return

    reader = get_video_reader(video_path)
    writer = get_writer(cfg.video_path, fps=30, crf=CRF)
    for image in tqdm(reader, total=get_video_lwh(video_path)[0], desc="Copy"):
        writer.write_frame(image)
    writer.close()
    reader.close()


def _compose_demo_cfg(video_path: Path, output_root: Path, static_camera: bool, use_dpvo: bool, f_mm: int | None):
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    overrides = [
        f"video_name={video_path.stem}",
        f"output_root={output_root}",
        f"static_cam={static_camera}",
        f"use_dpvo={use_dpvo}",
        "verbose=False",
    ]
    if f_mm is not None:
        overrides.append(f"f_mm={f_mm}")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=overrides)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def _run_preprocess(cfg) -> dict:
    video_path = cfg.video_path
    paths = cfg.paths

    Log.info("[Preprocess] Start!")
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx from {paths.bbx}")

    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")

    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    if not cfg.static_cam and not Path(paths.slam).exists():
        if cfg.use_dpvo:
            if SLAMModel is None:
                raise RuntimeError("GVHMR DPVO/SLAM preprocessing is unavailable in this runtime.")

            length, width, height = get_video_lwh(cfg.video_path)
            intrinsics = convert_K_to_K4(estimate_K(width, height))
            slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
            while slam.track():
                pass
            torch.save(slam.process(), paths.slam)
        else:
            simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
            torch.save(simple_vo.compute(), paths.slam)
    elif not cfg.static_cam:
        Log.info(f"[Preprocess] slam results from {paths.slam}")

    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        r_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            r_w2c = quaternion_to_matrix(traj_quat).mT
        else:
            r_w2c = torch.from_numpy(traj[:, :3, :3])

    if cfg.f_mm is not None:
        k_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        k_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": k_fullimg,
        "cam_angvel": compute_cam_angvel(r_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    Log.info("[Preprocess] End.")
    return data


def run_gvhmr_no_render(
    video: str | Path,
    output_root: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
    static_camera: bool = True,
    use_dpvo: bool = False,
    f_mm: int | None = None,
) -> Path:
    video_path = Path(video).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    checkpoint_root = _resolve_checkpoint_root(checkpoint_root)
    cfg = _compose_demo_cfg(video_path, output_root, static_camera, use_dpvo, f_mm)
    cfg.ckpt_path = checkpoint_root / "gvhmr" / "gvhmr_siga24_release.ckpt"
    _copy_video_to_gvhmr_workspace(video_path, cfg)

    data = _run_preprocess(cfg)
    result_path = Path(cfg.paths.hmr4d_results)
    if not result_path.exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra_utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        start = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - start:.2f}s")
        torch.save(pred, result_path)
    else:
        Log.info(f"[HMR4D] Reuse {result_path}")

    return result_path


def _command_value(command_line: list[str], name: str, default: str | None = None) -> str | None:
    if name not in command_line:
        return default
    index = command_line.index(name)
    return command_line[index + 1]


def main() -> None:
    # 本文件由 video_to_human_motion.py 以独立子进程调用（GVHMR 显存较大，跑完即随进程释放），
    # 因此入口按子进程约定从命令行读取视频与输出路径，而不是像其它入口写成可改变量。
    command_line = sys.argv[1:]
    video = _command_value(command_line, "--video")
    output_root = _command_value(command_line, "--output-root")
    checkpoint_root = _command_value(command_line, "--checkpoint-root")
    f_mm_text = _command_value(command_line, "--f-mm")

    if video is None or output_root is None:
        raise SystemExit(
            "usage: run_gvhmr_no_render.py --video VIDEO --output-root DIR [-s]"
        )

    result = run_gvhmr_no_render(
        Path(video),
        Path(output_root),
        checkpoint_root=Path(checkpoint_root) if checkpoint_root is not None else None,
        static_camera="-s" in command_line or "--static-camera" in command_line,
        use_dpvo="--use-dpvo" in command_line,
        f_mm=int(f_mm_text) if f_mm_text is not None else None,
    )
    print(result)


if __name__ == "__main__":
    main()
