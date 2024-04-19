import bisect
from itertools import accumulate
import json
import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import Dataset

from typing import List
import torch
import numpy as np

# TODO: ask for jit, https://pytorch.org/vision/stable/transforms.html

class Threshold:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, x):
        return (x >= self.threshold).float()

class NormalizeTransform:
    def __init__(self, rgb=True, depth=True, thermal=True):
        self.mean_rgb = np.array([126.39476776123047,128.59066772460938,134.02708435058594])
        self.mean_depth = 2959.2861328125
        self.mean_thermal = 29903.11328125 
        self.std_rgb = np.array([84.279945,83.32872,82.45626])
        self.std_depth = 1928.0287860921578
        self.std_thermal = 149.91512572744287
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal

    def __call__(self, frames):
        tensors = []
        i = 0
        if self.rgb:  # RGB Image
            frame = (frames[i] - self.mean_rgb) / self.std_rgb
            frame = torch.from_numpy(frame).float().permute(2, 0, 1)
            tensors.append(frame)
            i+=1
        if self.depth:  # Depth Image
            frame = (frames[i] - self.mean_depth) / self.std_depth
            frame = torch.from_numpy(frame).float().unsqueeze(0)
            tensors.append(frame)
            i+=1
        if self.thermal:  # Thermal Image
            frame = (frames[i] - self.mean_thermal) / self.std_thermal
            frame = torch.from_numpy(frame).float().unsqueeze(0)
            tensors.append(frame)
            i+=1
        result = torch.cat(tensors, dim=0)
        return result


class NormalizeListTransform:
    def __init__(self, rgb=True, depth=True, thermal=True):
        self.normalize = NormalizeTransform(rgb, depth, thermal)
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal

    def __call__(self, all_frames):
        all_tensors = []
        for frames in all_frames:
            tensors = self.normalize(frames)
            all_tensors.append(tensors)
        all_tensors = torch.stack(all_tensors, dim=0)
        return all_tensors

class ActionTransform:

    def __init__(self):
        self.actions = [
            'put_down', 'pick_up', 'drink', 'type', 'wave',
            'get_down', 'get_up',
            'sit', 'walk', 'stand', 'lay',
            'out_of_view', 'out_of_room', 'in_room'
        ]

    def __call__(self, y: List[np.ndarray]):
        labels = np.array([label in y[0] for label in self.actions])
        return torch.from_numpy(labels.astype(np.float32))


class MaskListTransform:
    def __call__(self, y: List[np.ndarray]):
        mask = y[0]
        return torch.from_numpy(mask.astype(np.float32))


class ActionListTransform:

    def __init__(self, selected_actions):
        self.actions = selected_actions
        self.transform = ActionTransform()

    def __call__(self, y: List[List[np.ndarray]]):
        all_labels = y[1]
        all_labels = np.array([
            np.array([
                label in labels
                for label in self.actions
            ]) for labels in all_labels]
        )
        all_labels = np.any(all_labels, axis=0)
        return torch.from_numpy(all_labels.astype(np.float32))

class MultiModalShotDataset(Dataset):

    def __init__(self,
                 root,
                 modalities: List[str] = ['rgb', 'depth', 'thermal'],
                 transform=None,
                 target_transform=None,
                 window_size=8) -> None:

        self.modalities = modalities
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size

        files = {}
        last_modality = None

        for modality in modalities:
            sample_path = os.path.join(root, modality)
            files[modality] = sorted(
                glob.glob(os.path.join(sample_path, '*')))
            if last_modality and files[modality] != files[last_modality]:
                raise Exception(f'ambitious data in {root}')

        self.files = files
        self.read_frame = {
            'rgb': lambda idx: cv2.imread(self.files['rgb'][idx], cv2.IMREAD_COLOR)[..., ::-1],
            'depth': lambda idx: cv2.imread(self.files['depth'][idx], cv2.IMREAD_ANYDEPTH),
            'thermal': lambda idx: cv2.imread(self.files['thermal'][idx], cv2.IMREAD_ANYDEPTH),
        }

        with open(os.path.join(root, 'actions.txt'), 'r') as f:
            self.actions = [int(line)
                            for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.files[self.modalities[0]])

    def __getitem__(self, idx: int):
        if self.window_size == 1:
            frames = [
                self.read_frame[modality](idx)
                for modality in self.modalities
            ]
            label = self.actions[idx]
            if self.transform != None:
                frames = self.transform(frames)
            if self.target_transform != None:
                label = self.target_transform(label)
            return frames, label
        else:
            all_frames = []
            actions = []
            for i in range(idx, idx+self.window_size):
                frames = [
                    self.read_frame[modality](i)
                    for modality in self.modalities
                ]
                all_frames.append(frames)
                # TODO: fix
                if i < len(self.actions) and i >= 0:
                    actions.append(self.actions[i])
                else:
                    actions.append([''])
            label = actions
            if self.transform != None:
                all_frames = self.transform(all_frames)
            if self.target_transform != None:
                label = self.target_transform(label)
            return all_frames, label


class MultiModalDataset(Dataset):

    def __init__(self,
                 root='../data',
                 split='train',
                 rgb=True, depth=True, thermal=True,
                 transform=None,
                 target_transform=None,
                 window_size=1):
        modalities: List[str] = []
        if rgb:
            modalities.append('rgb')
        if depth:
            modalities.append('depth')
        if thermal:
            modalities.append('thermal')
        assert all([modality in ['rgb', 'depth', 'thermal']
                   for modality in modalities])
        assert split in ['train', 'val', 'test']
        shots_dir = root
        shots_dir = os.path.join(shots_dir, split)
        shot_dirs = [
            os.path.join(shots_dir, str(i))
            for i in os.listdir(shots_dir)
        ]
        self.shots = [
            MultiModalShotDataset(
                shot_dir, modalities, transform, target_transform, window_size
            )
            for shot_dir in shot_dirs
        ]
        self.window_size = window_size
        self.cum_sum = [0] + list(accumulate(
            len(shot) - self.window_size + 1 for shot in self.shots))

    def __len__(self) -> int:
        return self.cum_sum[-1]

    def __getitem__(self, idx: int):
        array_index = bisect.bisect_left(self.cum_sum, idx + 1)
        return self.shots[array_index-1][idx-self.cum_sum[array_index-1]]