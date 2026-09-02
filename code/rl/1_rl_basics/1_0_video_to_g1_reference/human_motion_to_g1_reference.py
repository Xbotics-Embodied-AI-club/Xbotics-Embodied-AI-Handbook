"""管线第二步：把人体动作重定向成宇树 G1 的参考轨迹。

调用 GMR：先对齐人与机器人的关键部位、按体型做非均匀缩放，再逐帧解带约束的
逆运动学，输出 G1 真正能执行的关节角序列。

讲义对应：第14讲 4.3 节（GMR）与 4.4 节（四步管线）。
"""
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
    """列出这一步额外需要的第三方包。

    Returns:
        包名列表。
    """
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
    """定位一份完整的 GMR 源码目录。

    只认「资产和 IK 配置都在」的目录，避免拿到装了一半的包之后在半路才失败。

    Args:
        source_root: 显式指定的目录；给 None 时自动依次尝试已安装位置与下载目录。

    Returns:
        GMR 源码根目录。

    Raises:
        FileNotFoundError: 显式指定的目录不完整。
        RuntimeError: 自动查找失败。
    """
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
    """给出 GMR 资产的默认下载目录。

    Returns:
        下载目录路径。
    """
    return Path(os.environ["DATASETS_ROOT"]) / "models" / "downloaded" / "gmr"


def find_smplx_folder(smplx_folder: str | Path | None = None) -> Path:
    """定位 SMPL-X 人体模型文件所在目录。

    Args:
        smplx_folder: 显式指定的目录；给 None 时用 GMR 自带的位置。

    Returns:
        含 SMPL-X 模型的目录。

    Raises:
        RuntimeError: 找不到中性体型模型文件。
    """
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
    """开跑之前先确认 GMR 的机器人模型、IK 配置、人体模型都在。

    重定向要跑很久，缺件却要到中途才暴露；一次性把三样都查了，缺什么一起报出来。

    Args:
        robot: 目标机器人名。
        smplx_folder: SMPL-X 模型目录，给 None 时自动定位。

    Raises:
        RuntimeError: 有任何一件缺失，报出全部缺失项。
    """
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
    """把 GVHMR 的人体动作逐帧重定向成机器人关节角。

    Args:
        gvhmr_prediction: GVHMR 预测文件。
        target_fps: 输出帧率。
        robot: 目标机器人名。
        smplx_folder: SMPL-X 模型目录。

    Returns:
        (关节角序列, 实际帧率)。
    """
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
    """把重定向结果按 GMR 自己的格式存一份。

    Args:
        qpos: 关节角序列。
        fps: 帧率。
        output_path: 落盘路径。

    Returns:
        实际写出的文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(_motion_dict_from_qpos(qpos, fps), handle)
    return output_path


def write_gmr_csv_from_pickle(pickle_path: str | Path, csv_path: str | Path | None = None) -> Path:
    """把 GMR 的 pickle 结果转成 CSV，供下一步用 mjlab 回放。

    Args:
        pickle_path: 输入 pickle。
        csv_path: 输出 CSV；给 None 时按 pickle 的位置自动拼一个。

    Returns:
        实际写出的 CSV 路径。
    """
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
    """跑完整的重定向：读预测、查资产、解 IK、落盘。

    Args:
        gvhmr_prediction: GVHMR 预测文件。
        output_path: 结果落盘路径。

    Returns:
        重定向结果的路径。
    """
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
    """对上一步恢复出的人体动作跑一次重定向。"""
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
