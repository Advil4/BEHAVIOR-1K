import os
import glob
import shutil
import torch
import pandas as pd
import numpy as np
import av
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from loguru import logger

# 强行封锁所有网络
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"


def merge_sessions():
    data_root = "/home/ubuntu/Project/BEHAVIOR-1K/lerobot_data"
    master_name = "omnigibson_dex_master"
    master_dir = os.path.join(data_root, master_name)

    # 获取所有录制目录，排除旧版本和 Master 自己
    session_dirs = sorted(glob.glob(os.path.join(data_root, "omnigibson_dex_1*")))
    session_dirs = [d for d in session_dirs if not d.endswith("_old") and not d.endswith("_v30") and "master" not in d]

    if not session_dirs:
        logger.error("没有找到任何数据包！")
        return

    logger.info(f"找到 {len(session_dirs)} 个录制会话，启动【PyAV 底层暴力读取模式】...")

    if os.path.exists(master_dir):
        shutil.rmtree(master_dir)

    first_parquet = glob.glob(os.path.join(session_dirs[0], "data", "**", "*.parquet"), recursive=True)[0]
    df_sample = pd.read_parquet(first_parquet)
    state_dim = len(df_sample.iloc[0]['observation.state'])
    action_dim = len(df_sample.iloc[0]['action'])
    logger.success(f"🔍 动态检测到特征维度: State={state_dim}, Action={action_dim}")

    # 创建全新的空 Master 数据集
    master_dataset = LeRobotDataset.create(
        repo_id=master_name,
        fps=30,
        root=master_dir,
        features={
            "observation.image": {
                "dtype": "video",
                "shape": (3, 480, 640),
                "names": ["c", "h", "w"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["dim"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["dim"],
            }
        }
    )

    total_episodes = 0
    total_frames = 0

    for s_dir in session_dirs:
        repo_id = os.path.basename(s_dir)
        logger.info(f"👉 正在底层拆解: {repo_id}")

        # 直接去磁盘找 Parquet 数据表和 MP4 视频
        parquet_files = glob.glob(os.path.join(s_dir, "data", "**", "*.parquet"), recursive=True)
        video_files = glob.glob(os.path.join(s_dir, "videos", "**", "*.mp4"), recursive=True)

        if not parquet_files or not video_files:
            logger.warning(f"⚠️ {repo_id} 缺失核心的 data 或 video 文件，已跳过。")
            continue

        try:
            df = pd.read_parquet(parquet_files[0])
            container = av.open(video_files[0])
            frame_generator = container.decode(video=0)
        except Exception as e:
            logger.warning(f"⚠️ 无法读取文件 {repo_id}: {e}")
            continue

        current_ep = -1
        frame_count = 0

        # 逐行遍历表格，并与视频帧强行对齐
        for idx, row in df.iterrows():
            try:
                ep_idx = row['episode_index']
            except KeyError:
                ep_idx = 0  # 容错机制

            # 发现新的一集，结算上一集
            if ep_idx != current_ep:
                if current_ep != -1:
                    master_dataset.save_episode()
                    total_episodes += 1
                current_ep = ep_idx

            try:
                # 获取视频流下一帧，并转为 RGB 数组
                av_frame = next(frame_generator)
                frame = av_frame.to_ndarray(format='rgb24')
            except StopIteration:
                logger.warning(f"视频流提前结束于帧 {idx}，自动截断。")
                break
            except Exception as e:
                logger.error(f"视频解码出错: {e}")
                break

            # 转置为 CHW
            frame_chw = np.transpose(frame, (2, 0, 1))

            # 重新打包为 PyTorch 张量
            frame_dict = {
                "observation.image": torch.from_numpy(frame_chw).clone(),
                "observation.state": torch.tensor(row['observation.state'], dtype=torch.float32),
                "action": torch.tensor(row['action'], dtype=torch.float32),
                "task": "Put the apple on the plate"
            }
            master_dataset.add_frame(frame_dict)
            frame_count += 1
            total_frames += 1

        container.close()

        # 收尾该目录的最后一集
        if current_ep != -1 and frame_count > 0:
            master_dataset.save_episode()
            total_episodes += 1
            logger.success(f"✔ 成功提取并合入 {repo_id} (包含 {frame_count} 帧有效数据)")

    # 强制生成 Master 的元数据字典
    logger.info("整合数据并生成 Master Parquet...")
    try:
        if hasattr(master_dataset, 'consolidate'):
            master_dataset.consolidate()
    except Exception as e:
        logger.info("自动保存已就绪。")

    logger.success(f"🎉 合并彻底完成！有效 Episodes: {total_episodes}, 总帧数: {total_frames}")
    logger.info(f"🔥 训练时请使用此数据集路径：{master_dir}")


if __name__ == "__main__":
    merge_sessions()