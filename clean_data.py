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


def clean_dataset():
    data_root = "/home/ubuntu/Project/BEHAVIOR-1K/lerobot_data"

    # 【请替换为你真实的包含错数据的文件夹名】
    source_name = "/home/ubuntu/Project/BEHAVIOR-1K/lerobot_data/omnigibson_dex_1779870215"  # <--- 填入你的原数据集名字
    source_dir = os.path.join(data_root, source_name)

    # 清洗后的干净数据集
    clean_name = f"{source_name}_clean"
    clean_dir = os.path.join(data_root, clean_name)

    # ⚠️ 剔除的黑名单
    BAD_EPISODES = [43]

    if not os.path.exists(source_dir):
        logger.error(f"找不到源数据集: {source_dir}")
        return

    logger.info(f"启动【底层暴力清洗模式】，准备过滤的 Episode 黑名单: {BAD_EPISODES}")

    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)

    # 获取底层 Parquet 和 视频
    parquet_files = glob.glob(os.path.join(source_dir, "data", "**", "*.parquet"), recursive=True)
    video_files = glob.glob(os.path.join(source_dir, "videos", "**", "*.mp4"), recursive=True)

    if not parquet_files or not video_files:
        logger.error(f"源数据集 {source_name} 缺失核心的 data 或 video 文件！")
        return

    # 获取维度信息
    df = pd.read_parquet(parquet_files[0])
    state_dim = len(df.iloc[0]['observation.state'])
    action_dim = len(df.iloc[0]['action'])
    logger.success(f"🔍 检测到特征维度: State={state_dim}, Action={action_dim}")

    # 创建全新的空数据集
    clean_dataset = LeRobotDataset.create(
        repo_id=clean_name,
        fps=30,
        root=clean_dir,
        features={
            "observation.image": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["dim"]},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": ["dim"]}
        }
    )

    try:
        container = av.open(video_files[0])
        frame_generator = container.decode(video=0)
    except Exception as e:
        logger.error(f"无法打开源视频文件: {e}")
        return

    total_episodes_saved = 0
    total_frames_saved = 0

    current_ep = -1
    ep_frame_count = 0
    skipped_episodes_logged = set()

    # 逐行遍历表格，并与视频帧强行对齐
    for idx, row in df.iterrows():
        try:
            ep_idx = row['episode_index']
        except KeyError:
            ep_idx = 0

            # 发现新的一集，结算上一集
        if ep_idx != current_ep:
            # 只有当上一集【不是坏数据】且【有有效帧】时，才执行 save_episode()
            if current_ep != -1 and current_ep not in BAD_EPISODES and ep_frame_count > 0:
                clean_dataset.save_episode()
                total_episodes_saved += 1
                logger.info(f"✅ Episode {current_ep} 搬运完成。")

            current_ep = ep_idx
            ep_frame_count = 0

        try:
            av_frame = next(frame_generator)
        except StopIteration:
            logger.warning(f"视频流提前结束于帧 {idx}，自动截断。")
            break
        except Exception as e:
            logger.error(f"视频解码出错: {e}")
            break

        if ep_idx in BAD_EPISODES:
            if ep_idx not in skipped_episodes_logged:
                logger.warning(f"🚫 检测到黑名单数据 Episode {ep_idx}，正在抛弃该集所有帧...")
                skipped_episodes_logged.add(ep_idx)
            continue  # 跳过本帧的后续处理和保存逻辑

        # 如果是好的数据，正常转 RGB、CHW 并封装
        frame = av_frame.to_ndarray(format='rgb24')
        frame_chw = np.transpose(frame, (2, 0, 1))

        frame_dict = {
            "observation.image": torch.from_numpy(frame_chw).clone(),
            "observation.state": torch.tensor(row['observation.state'], dtype=torch.float32),
            "action": torch.tensor(row['action'], dtype=torch.float32),
            "task": "Put the apple on the plate"
        }
        clean_dataset.add_frame(frame_dict)

        ep_frame_count += 1
        total_frames_saved += 1

    container.close()

    # 收尾最后一集
    if current_ep != -1 and current_ep not in BAD_EPISODES and ep_frame_count > 0:
        clean_dataset.save_episode()
        total_episodes_saved += 1
        logger.info(f"✅ Episode {current_ep} 搬运完成。")

    # 封存合并
    logger.info("生成最终数据字典...")
    try:
        if hasattr(clean_dataset, 'consolidate'):
            clean_dataset.consolidate()
    except Exception as e:
        logger.info("自动保存已就绪。")

    logger.success(f"🎉 清洗彻底完成！保留 Episodes: {total_episodes_saved}, 保留总帧数: {total_frames_saved}")
    logger.info(f"🔥 干净的数据集路径：{clean_dir}")


if __name__ == "__main__":
    clean_dataset()
