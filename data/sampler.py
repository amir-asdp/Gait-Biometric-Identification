"""
Custom samplers for gait recognition training.

Implements batch samplers that ensure each batch contains multiple samples
from the same subjects for effective triplet loss training.
"""

import random
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class TripletSampler(Sampler):
    """
    Sampler for triplet loss training.
    
    This sampler ensures each batch contains:
    - P different persons (identities)
    - K samples per person
    
    Total batch size = P * K
    
    This structure is essential for batch-hard triplet mining where we need
    multiple samples from the same identity in each batch.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    batch_size : int
        Total batch size (P * K).
    person_num : int
        Number of different persons per batch (P).
    sample_num : int
        Number of samples per person (K).
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 128,
        person_num: int = 8,
        sample_num: int = 16,
    ):
        """Initialize triplet sampler."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.person_num = person_num
        self.sample_num = sample_num

        # Verify batch size
        assert batch_size == person_num * sample_num, \
            f"batch_size ({batch_size}) must equal person_num ({person_num}) * sample_num ({sample_num})"
        
        # Build subject-to-samples mapping
        self.subject_to_samples = self._build_subject_mapping()
        
        # Get list of subjects
        self.subjects = list(self.subject_to_samples.keys())
        
        # Compute number of batches
        self.num_batches = len(self.subjects) // self.person_num
    
    def _build_subject_mapping(self) -> dict:
        """
        Build mapping from subject IDs to sample indices.
        
        Returns
        -------
        dict
            Dictionary mapping subject_id -> list of sample indices.
        """
        subject_to_samples = {}
        
        for idx in range(len(self.dataset)):
            subject_id = self.dataset.data_index[idx]['subject_id']
            
            if subject_id not in subject_to_samples:
                subject_to_samples[subject_id] = []
            
            subject_to_samples[subject_id].append(idx)
        
        return subject_to_samples
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate batches for one epoch.
        
        Yields
        ------
        int
            Sample index.
        """
        # Shuffle subjects
        subjects = self.subjects.copy()
        random.shuffle(subjects)
        
        # Generate batches
        for batch_idx in range(self.num_batches):
            # Select P persons for this batch
            batch_subjects = subjects[batch_idx * self.person_num:(batch_idx + 1) * self.person_num]
            
            batch_indices = []
            
            for subject_id in batch_subjects:
                # Get all samples for this subject
                subject_samples = self.subject_to_samples[subject_id]
                
                # Sample K samples (with replacement if needed)
                if len(subject_samples) >= self.sample_num:
                    selected = random.sample(subject_samples, self.sample_num)
                else:
                    # If not enough samples, sample with replacement
                    selected = random.choices(subject_samples, k=self.sample_num)
                
                batch_indices.extend(selected)
            
            # Yield indices in this batch
            for idx in batch_indices:
                yield idx
    
    def __len__(self) -> int:
        """
        Return the total number of samples per epoch.
        
        Returns
        -------
        int
            Number of samples.
        """
        return self.num_batches * self.batch_size


class BalancedSampler(Sampler):
    """
    Balanced sampler that ensures equal representation of all subjects.
    
    This sampler is useful for evaluation to ensure fair comparison across subjects.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    samples_per_subject : int
        Number of samples to draw from each subject.
    """
    
    def __init__(self, dataset, samples_per_subject: int = 10):
        """Initialize balanced sampler."""
        self.dataset = dataset
        self.samples_per_subject = samples_per_subject
        
        # Build subject-to-samples mapping
        self.subject_to_samples = self._build_subject_mapping()
        self.subjects = list(self.subject_to_samples.keys())
    
    def _build_subject_mapping(self) -> dict:
        """Build mapping from subject IDs to sample indices."""
        subject_to_samples = {}
        
        for idx in range(len(self.dataset)):
            subject_id = self.dataset.data_index[idx]['subject_id']
            
            if subject_id not in subject_to_samples:
                subject_to_samples[subject_id] = []
            
            subject_to_samples[subject_id].append(idx)
        
        return subject_to_samples
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate samples with balanced subject representation.
        
        Yields
        ------
        int
            Sample index.
        """
        all_indices = []
        
        for subject_id in self.subjects:
            subject_samples = self.subject_to_samples[subject_id]
            
            # Sample uniformly
            if len(subject_samples) >= self.samples_per_subject:
                selected = random.sample(subject_samples, self.samples_per_subject)
            else:
                selected = subject_samples * (self.samples_per_subject // len(subject_samples))
                selected += random.sample(subject_samples, self.samples_per_subject % len(subject_samples))
            
            all_indices.extend(selected)
        
        # Shuffle all indices
        random.shuffle(all_indices)
        
        for idx in all_indices:
            yield idx
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.subjects) * self.samples_per_subject

