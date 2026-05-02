"""
CASIA-B Dataset Loader for Gait Recognition.

This module implements a PyTorch Dataset for the CASIA-B gait dataset, supporting:
- Multiple conditions (normal, bag, clothing)
- Multiple view angles
- Gallery-Probe split for evaluation
- Data caching for faster training
"""

import os
import json
import pickle
import random
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import GaitTransform
from .sampler import TripletSampler


class CASIABDataset(Dataset):
    """
    CASIA-B Gait Dataset.
    
    The CASIA-B dataset contains 124 subjects with 10 sequences per subject under 3 conditions:
    - Normal walking (nm-01 to nm-06)
    - Walking with bag (bg-01, bg-02)
    - Walking with coat (cl-01, cl-02)
    
    Each sequence is captured from 11 view angles: 0°, 18°, 36°, 54°, 72°, 90°, 108°, 126°, 144°, 162°, 180°.
    
    Parameters
    ----------
    data_root : str
        Root directory of CASIA-B dataset.
    subjects : List[int]
        List of subject IDs to include.
    conditions : List[str]
        List of conditions to include (e.g., ['nm-01', 'nm-02']).
    views : Union[str, List[int]]
        View angles to include. 'all' for all views or list of view angles.
    frame_num : int
        Number of frames to sample per sequence.
    sample_type : str
        Sampling strategy: 'fixed' (uniform sampling) or 'unfixed' (random sampling).
    transform : Optional[GaitTransform]
        Transform to apply to the data.
    cache : bool
        Whether to cache data in memory for faster training.
    """
    
    # View angle mapping (folder name -> angle in degrees)
    VIEW_ANGLES = {
        '000': 0, '018': 18, '036': 36, '054': 54, '072': 72, '090': 90,
        '108': 108, '126': 126, '144': 144, '162': 162, '180': 180
    }
    
    def __init__(
        self,
        dataset_tag: str,
        data_root: str,
        subjects: List[int],
        conditions: List[str],
        views: Union[str, List[int]] = 'all',
        frame_num: int = 30,
        sample_type: str = 'fixed',
        transform: Optional[GaitTransform] = None,
        cache: bool = False,
        truncate_threshold: int = 20,
    ):
        """Initialize CASIA-B dataset."""
        super().__init__()

        self.data_root_path = Path(data_root)
        self.subjects = subjects
        self.conditions = conditions
        self.frame_num = frame_num
        self.sample_type = sample_type
        self.transform = transform
        self.cache = cache
        self.truncate_threshold = truncate_threshold
        self.dataset_tag = dataset_tag
        
        # Process view angles
        if views == 'all':
            self.views = list(self.VIEW_ANGLES.keys())
        else:
            # Convert degree angles to folder names
            self.views = [f"{angle:03d}" for angle in views]
        
        # Build dataset index
        self.data_index = self._build_index()
        
        # Cache for storing loaded data
        self._cache = {} if cache else None
        
        print(f"CASIA-B Dataset initialized:")
        print(f"  - Subjects: {len(self.subjects)}")
        print(f"  - Conditions: {len(self.conditions)}")
        print(f"  - Views: {len(self.views)}")
        print(f"  - Total samples: {len(self.data_index)}\n")
    
    def _build_index(self) -> List[Dict]:
        """
        Build an index of all available data samples with caching support.
        
        Returns
        -------
        List[Dict]
            List of dictionaries containing sample information:
            - subject_id: int
            - condition: str
            - view: str
            - view_angle: int (in degrees)
            - pkl_path: Path
        """
        # Create cache file path
        cache_file = Path(f"{self.data_root_path}/index_cache_{self.dataset_tag}.json")

        # Try to load from cache
        if cache_file.exists():
            print(f"Loading index from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    index = json.load(f)
                print(f"Loaded {len(index)} samples from cache")
                return index
            except:
                print("Cache corrupted, rebuilding...")

        # Build index (use parallel version from Solution 1)
        index = self._build_index_parallel()  # Use parallel method

        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(index, f)
            print(f"Index cached to {cache_file}")
        except:
            print("Warning: Could not save cache")

        return index

    def _build_index_parallel(self) -> List[Dict]:
        """
        Build an index of all available data samples using parallel processing.
        Returns
        -------
        List[Dict]
            List of dictionaries containing sample information:
            - subject_id: int
            - condition: str
            - view: str
            - view_angle: int (in degrees)
            - pkl_path: Path
        """
        # Generate all combinations
        all_combinations = []
        for subject_id in self.subjects:
            for condition in self.conditions:
                for view in self.views:
                    all_combinations.append((subject_id, condition, view))

        # Function to check one path


        # Parallel execution with progress bar
        results = Parallel(n_jobs=64, backend='threading')(
            delayed(self._get_sample_info_if_exists)(subj, cond, view)
            for subj, cond, view in tqdm(all_combinations, desc=f"Building index for {self.dataset_tag}")
        )

        # Filter out None results
        index = [r for r in results if r is not None]

        return index

    def _get_sample_info_if_exists(self, subject_id, condition, view):
        """
        Get sample information of a gait sequence data if its file exists.
        Returns
        -------
        Dict
            A dictionary containing sample information:
            - subject_id: int
            - condition: str
            - view: str
            - view_angle: int (in degrees)
            - pkl_path: Path
        """
        subject_str = f"{subject_id:03d}"
        pkl_path = f"{self.data_root_path}/{subject_str}/{condition}/{view}/{view}-sils.pkl"

        if Path(pkl_path).exists():
            return {
                'subject_id': subject_id,
                'condition': condition,
                'view': view,
                'view_angle': self.VIEW_ANGLES[view],
                'pkl_path': pkl_path,
            }
        else:
            print(f"Warning: Subject {subject_str} not found at {pkl_path}")
            return None
    
    def _load_silhouettes(self, pkl_path: Path) -> np.ndarray:
        """
        Load silhouette sequence from pickle file.
        
        Parameters
        ----------
        pkl_path : Path
            Path to the pickle file.
        
        Returns
        -------
        np.ndarray
            Silhouette sequence of shape [T, H, W] where T is number of frames.
        """
        # Check cache first
        if self._cache is not None and str(pkl_path) in self._cache:
            return self._cache[str(pkl_path)]
        
        # Load from disk
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different possible data formats
        if isinstance(data, dict):
            # Try common keys
            for key in ['sils', 'silhouettes', 'data']:
                if key in data:
                    sils = data[key]
                    break
            else:
                sils = list(data.values())[0]
        elif isinstance(data, (list, tuple)):
            sils = np.array(data)
        else:
            sils = data
        
        # Ensure numpy array
        if not isinstance(sils, np.ndarray):
            sils = np.array(sils)
        
        # Ensure shape is [T, H, W]
        if sils.ndim == 4:  # [T, H, W, C]
            sils = sils.squeeze(-1)
        
        # Normalize to [0, 1]
        if sils.max() > 1.0:
            sils = sils.astype(np.float32) / 255.0
        
        # Cache if enabled
        if self._cache is not None:
            self._cache[str(pkl_path)] = sils
        
        return sils
    
    def _sample_frames(self, sils: np.ndarray) -> np.ndarray:
        """
        Sample frames from silhouette sequence.
        
        Parameters
        ----------
        sils : np.ndarray
            Full silhouette sequence of shape [T, H, W].
        
        Returns
        -------
        np.ndarray
            Sampled silhouette sequence of shape [frame_num, H, W].
        """
        num_frames = len(sils)
        
        # If sequence is too short, repeat frames
        if num_frames < self.truncate_threshold:
            # Repeat sequence to reach minimum length
            repeat_times = (self.truncate_threshold // num_frames) + 1
            sils = np.tile(sils, (repeat_times, 1, 1))
            num_frames = len(sils)
        
        if self.sample_type == 'fixed':
            # Uniform sampling
            if num_frames >= self.frame_num:
                indices = np.linspace(0, num_frames - 1, self.frame_num, dtype=int)
            else:
                # Pad with last frame if needed
                indices = list(range(num_frames))
                indices += [num_frames - 1] * (self.frame_num - num_frames)
                indices = np.array(indices)
        
        elif self.sample_type == 'unfixed':
            # Random sampling
            if num_frames >= self.frame_num:
                indices = sorted(random.sample(range(num_frames), self.frame_num))
            else:
                indices = list(range(num_frames))
                indices += random.choices(range(num_frames), k=self.frame_num - num_frames)
                indices = sorted(indices)
        
        else:
            raise ValueError(f"Unknown sample_type: {self.sample_type}")
        
        return sils[indices]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Sample index.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - silhouettes: Tensor of shape [frame_num, H, W]
            - subject_id: Subject ID (label)
            - view_angle: View angle in degrees
            - condition: Condition string
        """
        sample_info = self.data_index[idx]
        
        # Load silhouettes
        sils = self._load_silhouettes(sample_info['pkl_path'])
        
        # Sample frames
        sils = self._sample_frames(sils)
        
        # Convert to tensor [T, H, W]
        sils = torch.from_numpy(sils).float()
        
        # Apply transforms
        if self.transform is not None:
            sils = self.transform(sils)
        
        return {
            'silhouettes': sils,
            'subject_id': sample_info['subject_id'] - 1,  # Convert to 0-indexed
            'view_angle': sample_info['view_angle'],
            'condition': sample_info['condition'],
        }
    
    def get_subject_samples(self, subject_id: int) -> List[int]:
        """
        Get all sample indices for a given subject.
        
        Parameters
        ----------
        subject_id : int
            Subject ID.
        
        Returns
        -------
        List[int]
            List of sample indices.
        """
        return [i for i, sample in enumerate(self.data_index) 
                if sample['subject_id'] == subject_id]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Parameters
    ----------
    batch : List[Dict]
        List of samples from dataset.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Batched data.
    """
    silhouettes = torch.stack([sample['silhouettes'] for sample in batch])
    subject_ids = torch.tensor([sample['subject_id'] for sample in batch], dtype=torch.long)
    view_angles = torch.tensor([sample['view_angle'] for sample in batch], dtype=torch.long)
    
    return {
        'silhouettes': silhouettes,
        'subject_ids': subject_ids,
        'view_angles': view_angles,
    }


def get_dataloader(
    config: Dict,
    mode: str = 'train',
    sampler: Optional[TripletSampler] = None,
) -> DataLoader:
    """
    Create DataLoader for training or evaluation.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary.
    mode : str
        'train', 'gallery', or 'probe'.
    sampler : Optional[TripletSampler]
        Custom sampler for training.
    
    Returns
    -------
    DataLoader
        PyTorch DataLoader.
    """
    dataset_cfg = config['dataset']
    
    # Get subjects and conditions based on mode
    if mode == 'train':
        subjects = list(range(dataset_cfg['train']['subjects'][0], dataset_cfg['train']['subjects'][1] + 1))
        conditions = dataset_cfg['train']['conditions']
        views = 'all'
        shuffle = (sampler is None)
        batch_size = config['training']['batch_size']
    
    elif mode == 'gallery':
        subjects = list(range(dataset_cfg['gallery']['subjects'][0], dataset_cfg['gallery']['subjects'][1] + 1))
        conditions = dataset_cfg['gallery']['conditions']
        views = dataset_cfg['gallery']['views']
        shuffle = False
        batch_size = config['evaluation']['batch_size']
    
    elif mode == 'probe':
        subjects = list(range(dataset_cfg['probe']['subjects'][0], dataset_cfg['probe']['subjects'][1] + 1))
        # For probe, we might want to evaluate on different conditions separately
        # For now, include all probe conditions
        conditions = (dataset_cfg['probe']['nm'] + 
                     dataset_cfg['probe']['bg'] + 
                     dataset_cfg['probe']['cl'])
        views = dataset_cfg['probe']['views']
        shuffle = False
        batch_size = config['evaluation']['batch_size']
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Create transform
    if mode == 'train' and dataset_cfg['augmentation']['enabled']:
        transform = GaitTransform(
            resolution=tuple(dataset_cfg['input']['resolution']),
            horizontal_flip=dataset_cfg['augmentation']['horizontal_flip'],
            flip_prob=dataset_cfg['augmentation']['flip_prob'],
            random_rotation=dataset_cfg['augmentation']['random_rotation'],
            random_erasing=dataset_cfg['augmentation']['random_erasing'],
            erasing_prob=dataset_cfg['augmentation']['erasing_prob'],
        )
    else:
        transform = GaitTransform(
            resolution=tuple(dataset_cfg['input']['resolution']),
            horizontal_flip=False,
        )
    
    # Create dataset
    if sampler is None:
        dataset = CASIABDataset(
            data_root=dataset_cfg['data_root'],
            subjects=subjects,
            conditions=conditions,
            views=views,
            frame_num=dataset_cfg['input']['frame_num'],
            sample_type=dataset_cfg['input']['sample_type'],
            transform=transform,
            cache=dataset_cfg['cache_enabled'],
            truncate_threshold=dataset_cfg['input']['truncate_threshold'],
            dataset_tag=f"{mode}_data"
        )
    else:
        dataset = sampler.dataset
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config['device']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['device']['pin_memory'],
        persistent_workers=config['device']['persistent_workers'],
        drop_last=(mode == 'train'),
    )
    
    return dataloader

