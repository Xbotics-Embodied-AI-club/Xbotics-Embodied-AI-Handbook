from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting import motion_retarget, params
from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file


def required_packages() -> list[str]:
    return ["general_motion_retargeting"]


def _looks_like_gmr_source_root(path: Path) -> bool:
    return all(
        candidate.is_file()
        for candidate in [
            path / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
            path / "general_motion_retargeting" / "ik_configs" / "smplx_to_g1.json",
        ]
    )


def find_gmr_source_root(source_root: str | Path | None = None) -> Path:
    if source_root is not None:
        source_root = Path(source_root).expanduser().resolve()
        if not _looks_like_gmr_source_root(source_root):
            raise FileNotFoundError(f"GMR source root is incomplete: {source_root}")
        return source_root

    installed_root = Path(params.ASSET_ROOT).resolve().parent
    if _looks_like_gmr_source_root(installed_root):
        return installed_root

    downloaded_root = default_gmr_root()
    if _looks_like_gmr_source_root(downloaded_root):
        return downloaded_root

    raise RuntimeError(
        "GMR source assets are missing. Install general-motion-retargeting from GitHub with uv "
        "or put the downloaded asset bundle under DATASETS_ROOT/models/downloaded/gmr."
    )


def default_gmr_root() -> Path:
    return Path(os.environ["DATASETS_ROOT"]) / "models" / "downloaded" / "gmr"


def find_smplx_folder(smplx_folder: str | Path | None = None) -> Path:
    if smplx_folder is not None:
        smplx_folder = Path(smplx_folder)
    else:
        smplx_folder = _gmr_asset_root() / "body_models"

    neutral_model = smplx_folder / "smplx" / "SMPLX_NEUTRAL.npz"
    if not neutral_model.is_file():
        raise RuntimeError(
            "GMR SMPLX body model assets are missing. "
            f"Expected {neutral_model}. Install GMR with assets/body_models included."
        )
    return smplx_folder


def _gmr_asset_root():
    downloaded_asset_root = default_gmr_root() / "assets"
    if (downloaded_asset_root / "unitree_g1" / "g1_mocap_29dof.xml").is_file():
        return downloaded_asset_root

    if params is not None:
        installed_asset_root = Path(params.ASSET_ROOT)
        if (installed_asset_root / "unitree_g1" / "g1_mocap_29dof.xml").is_file():
            return installed_asset_root
    return find_gmr_source_root() / "assets"


def _gmr_ik_config_root():
    downloaded_ik_root = default_gmr_root() / "general_motion_retargeting" / "ik_configs"
    if (downloaded_ik_root / "smplx_to_g1.json").is_file():
        return downloaded_ik_root

    if params is not None:
        installed_ik_root = Path(params.IK_CONFIG_ROOT)
        if (installed_ik_root / "smplx_to_g1.json").is_file():
            return installed_ik_root
    return find_gmr_source_root() / "general_motion_retargeting" / "ik_configs"


def check_gmr_assets(robot: str = "unitree_g1", smplx_folder: str | Path | None = None) -> None:
    missing = []
    robot_xml = _gmr_asset_root() / "unitree_g1" / "g1_mocap_29dof.xml"
    if robot != "unitree_g1":
        robot_xml = Path(params.ROBOT_XML_DICT[robot])
    ik_config = _gmr_ik_config_root() / "smplx_to_g1.json"
    if robot != "unitree_g1":
        ik_config = Path(params.IK_CONFIG_DICT["smplx"][robot])
    if not robot_xml.is_file():
        missing.append(str(robot_xml))
    if not ik_config.is_file():
        missing.append(str(ik_config))
    try:
        find_smplx_folder(smplx_folder)
    except RuntimeError as exc:
        missing.append(str(exc))

    if missing:
        raise RuntimeError(
            "GMR package is incomplete for Unitree G1 retargeting. "
            "The installed package must include assets/ and general_motion_retargeting/ik_configs/. "
            f"Missing: {'; '.join(missing)}"
        )


def retarget_gvhmr_to_qpos(
    gvhmr_prediction: str | Path,
    target_fps: int,
    robot: str,
    *,
    smplx_folder: str | Path | None = None,
) -> tuple[np.ndarray, float]:
    gvhmr_prediction = Path(gvhmr_prediction)
    check_gmr_assets(robot, smplx_folder)

    params.ASSET_ROOT = _gmr_asset_root()
    params.ROBOT_XML_DICT["unitree_g1"] = params.ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml"
    motion_retarget.ROBOT_XML_DICT["unitree_g1"] = params.ROBOT_XML_DICT["unitree_g1"]
    params.IK_CONFIG_ROOT = _gmr_ik_config_root()
    params.IK_CONFIG_DICT["smplx"]["unitree_g1"] = params.IK_CONFIG_ROOT / "smplx_to_g1.json"
    motion_retarget.IK_CONFIG_DICT["smplx"]["unitree_g1"] = params.IK_CONFIG_DICT["smplx"]["unitree_g1"]

    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        gvhmr_prediction,
        find_smplx_folder(smplx_folder),
    )
    smplx_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=target_fps,
    )
    retarget = GeneralMotionRetargeting(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
    )

    qpos = [retarget.retarget(frame) for frame in smplx_frames]
    return np.asarray(qpos, dtype=np.float32), float(aligned_fps)


def _motion_dict_from_qpos(qpos: np.ndarray, fps: float) -> dict[str, object]:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"unitree_g1 qpos must have shape (T, 36), got {qpos.shape}")
    return {
        "fps": float(fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }


def write_gmr_pickle(qpos: np.ndarray, fps: float, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(_motion_dict_from_qpos(qpos, fps), handle)
    return output_path


def write_gmr_csv_from_pickle(pickle_path: str | Path, csv_path: str | Path | None = None) -> Path:
    pickle_path = Path(pickle_path)
    if csv_path is None:
        csv_path = pickle_path.parent / "csv" / pickle_path.with_suffix(".csv").name
    csv_path = Path(csv_path)

    with pickle_path.open("rb") as handle:
        motion_data = pickle.load(handle)

    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
    motion[:, :3] = np.asarray(motion_data["root_pos"], dtype=np.float32)
    motion[:, 3:7] = np.asarray(motion_data["root_rot"], dtype=np.float32)
    motion[:, 7:] = dof_pos

    frame_rate = float(motion_data["fps"])
    if frame_rate > 30:
        indices = np.arange(0, motion.shape[0], frame_rate / 30.0).astype(int)
        motion = motion[indices]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, motion, delimiter=",")
    return csv_path


def retarget_human_motion(
    gvhmr_prediction: str | Path,
    output_path: str | Path,
    *,
    robot: str = "unitree_g1",
    target_fps: int = 30,
    smplx_folder: str | Path | None = None,
    retargeter: Callable[[str | Path, int, str], tuple[np.ndarray, float]] | None = None,
) -> Path:
    gvhmr_prediction = Path(gvhmr_prediction)
    if not gvhmr_prediction.is_file():
        raise FileNotFoundError(f"GVHMR prediction not found: {gvhmr_prediction}")

    if retargeter is None:
        retargeter = lambda path, fps, bot: retarget_gvhmr_to_qpos(
            path,
            fps,
            bot,
            smplx_folder=smplx_folder,
        )

    qpos, aligned_fps = retargeter(gvhmr_prediction, target_fps, robot)
    output_path = write_gmr_pickle(qpos, aligned_fps, output_path)
    write_gmr_csv_from_pickle(output_path)
    return output_path


def main() -> None:
    # 主要修改这一段：GVHMR 预测文件、输出 pkl、目标机器人与帧率。
    gvhmr_prediction = Path("hmr4d_results.pt")
    output_pkl = Path("unitree_g1_motion.pkl")
    robot = "unitree_g1"
    target_fps = 30
    smplx_folder = None

    result = retarget_human_motion(
        gvhmr_prediction,
        output_pkl,
        robot=robot,
        target_fps=target_fps,
        smplx_folder=smplx_folder,
    )
    print(result)


if __name__ == "__main__":
    main()
