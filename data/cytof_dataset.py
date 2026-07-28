####################################################################################################
# 🎶 MAESTRO - MAsked Encoding Set TRansformer w/ self-DistillatiOn 🎶
# Author: Matthew E. Lee
# Advisors: E. John Wherry & Dokyoon Kim
# Contact: matthew.lee1@pennmedicine.upenn.edu
# cytof_dataset.py
####################################################################################################

import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_SYNONYMS = {
    'CD8a': 'CD8',
    'CD8b': 'CD8',
    'PD1': 'PD-1',
    'PDL1': 'PD-L1',
    'PD-L1': 'PD-L1',
    'HLADR': 'HLA-DR',
    'HLA_DR': 'HLA-DR',
    'GzmB': 'GranzymeB',
    'GZMB': 'GranzymeB',
    'KI67': 'Ki-67',
    'Ki67': 'Ki-67',
    'CTLA4': 'CTLA-4',
    'gdTCR': 'TCRgd',
    'PANGT': 'TCRgd',
    'FOXP3': 'FOXP3',
    'Tbet': 'T-bet',
    'IFNG': 'IFN-g',
    'TNF': 'TNF-a',
    'IL2': 'IL-2',
    'IL17': 'IL-17',
    'CCR7': 'CD197',
    'CCR4': 'CCR4',
    'CXCR3': 'CXCR3',
    'GranzymeB': 'GranzymeB',
    'TCR_Va24-Ja18': 'Va24-Ja18',
    'FcER1': 'FceR1',
    'FceR1': 'FceR1',
    'CXCR3': 'CD183',
    'CXCR5': 'CD185',
    'CCR4': 'CD194',
    'CCR6': 'CD196',
    'FceRI': 'FceR1',
    'CRTH2': 'CD294',
}

def canonicalize_marker(name):
    """Map a raw marker name to its canonical form via FEATURE_SYNONYMS."""
    return FEATURE_SYNONYMS.get(name, name)

class CyTOFDataset(Dataset):
    def __init__(self, data_dirs, subset_size=None, cell_type_removal=None, marker_dirs=None):
        """
        Args:
            data_dirs: str or list of str — one or more directories containing .h5 files (used for training)
            subset_size: int or None — if set, subsample/oversample each sample to this size
            cell_type_removal: list of str or None — cell types to exclude from each sample
            marker_dirs: str or list of str or None — additional directories whose panels are included
                         in the shared-marker intersection but whose data is NOT loaded for training
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if isinstance(marker_dirs, str):
            marker_dirs = [marker_dirs]

        self.data_dirs = data_dirs
        self.marker_dirs = marker_dirs or []
        self.subset_size = subset_size
        self.cell_type_removal = set(cell_type_removal) if cell_type_removal else set()
        self.file_paths, self.file_to_dir = self._get_file_paths()
        self.sample_names = list(self.file_paths.keys())
        self.cell_type_mapping = {}
        self.reverse_mapping = {}
        self.discovered_types = set()

        # Discover per-directory panels and compute shared markers
        self.shared_markers, self.dir_col_indices = self._build_shared_panel()

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank == 0:
            print(f"Initialized dataset: {len(self.sample_names)} samples from {len(data_dirs)} training directories")
            if self.marker_dirs:
                print(f"Marker-only directories ({len(self.marker_dirs)}): {self.marker_dirs}")
            print(f"Shared markers ({len(self.shared_markers)}): {self.shared_markers}")
            if self.cell_type_removal:
                print(f"Removing cell types: {sorted(self.cell_type_removal)}")

    def _get_file_paths(self):
        file_paths = {}
        file_to_dir = {}
        for data_dir in self.data_dirs:
            for filename in os.listdir(data_dir):
                if filename.endswith('.h5'):
                    sample_name = filename.replace('.h5', '')
                    if sample_name in file_paths:
                        raise ValueError(f"Duplicate sample name '{sample_name}' found in {data_dir}")
                    file_paths[sample_name] = os.path.join(data_dir, filename)
                    file_to_dir[sample_name] = data_dir
        return file_paths, file_to_dir

    def _build_shared_panel(self):
        """Read feature_names from one file per directory, canonicalize via FEATURE_SYNONYMS,
        compute intersection across training dirs and marker-only dirs, and precompute
        per-directory column indices (for training dirs only) into the shared canonical order."""
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dir_canon_panels = {}  # data_dir -> list of canonicalized marker names

        # Read panels from all directories (training + marker-only)
        all_dirs = self.data_dirs + self.marker_dirs
        for data_dir in all_dirs:
            is_marker_only = data_dir in self.marker_dirs
            for filename in os.listdir(data_dir):
                if filename.endswith('.h5'):
                    with h5py.File(os.path.join(data_dir, filename), 'r') as h5file:
                        raw = [m.decode('utf-8') for m in h5file['feature_names'][:]]
                    canon = [canonicalize_marker(m) for m in raw]
                    dir_canon_panels[data_dir] = canon
                    if local_rank == 0:
                        dir_name = os.path.basename(os.path.normpath(data_dir))
                        tag = " [marker-only]" if is_marker_only else ""
                        renamed = [(r, c) for r, c in zip(raw, canon) if r != c]
                        print(f"[Panel] {dir_name}{tag}: {len(raw)} markers, {len(renamed)} renamed")
                        for r, c in renamed:
                            print(f"[Panel]   {r} -> {c}")
                    break

        # Intersection over canonicalized names across all directories
        shared = set(dir_canon_panels[all_dirs[0]])
        for panel in dir_canon_panels.values():
            shared &= set(panel)
        shared_markers = sorted(shared)

        # Per-directory column indices — only needed for training dirs (used in __getitem__)
        dir_col_indices = {}
        for data_dir in self.data_dirs:
            canon_panel = dir_canon_panels[data_dir]
            canon_to_idx = {}
            for i, c in enumerate(canon_panel):
                if c not in canon_to_idx:  # first occurrence wins (shouldn't have dupes)
                    canon_to_idx[c] = i
            dir_col_indices[data_dir] = [canon_to_idx[m] for m in shared_markers]

        if local_rank == 0:
            print(f"[Panel] Shared markers ({len(shared_markers)}): {shared_markers}")

        return shared_markers, dir_col_indices

    ZERO_THRESHOLD = 1e-6

    def _register_cell_type(self, cell_type):
        """Register a new cell type using deterministic alphabetical ordering."""
        self.discovered_types.add(cell_type)
        sorted_types = sorted(self.discovered_types)
        self.cell_type_mapping = {ct: i for i, ct in enumerate(sorted_types)}
        self.reverse_mapping = {i: ct for i, ct in enumerate(sorted_types)}

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        file_path = self.file_paths[sample_name]

        with h5py.File(file_path, 'r') as h5file:
            raw_features = torch.tensor(h5file['data'][:])
            cell_types_str = [ct.decode('utf-8') for ct in h5file['cell_types'][:]]

        # Select only shared markers in canonical order
        col_indices = self.dir_col_indices[self.file_to_dir[sample_name]]
        features = raw_features[:, col_indices]

        # Filter out unwanted cell types
        if self.cell_type_removal:
            keep_mask = torch.tensor([ct not in self.cell_type_removal for ct in cell_types_str], dtype=torch.bool)
            features = features[keep_mask]
            cell_types_str = [ct for ct in cell_types_str if ct not in self.cell_type_removal]

        for ct in cell_types_str:
            if ct not in self.discovered_types:
                self._register_cell_type(ct)
        cell_types = torch.tensor([self.cell_type_mapping[ct] for ct in cell_types_str], dtype=torch.long)

        # Subsample or oversample to fixed size
        if self.subset_size is not None:
            n = features.shape[0]
            if n > self.subset_size:
                indices = torch.randperm(n)[:self.subset_size]
            elif n < self.subset_size:
                indices = torch.randint(0, n, (self.subset_size,))
            else:
                indices = torch.arange(n)
            features = features[indices]
            cell_types = cell_types[indices]

        return features, cell_types, sample_name

    def get_cell_type_name(self, idx):
        return self.reverse_mapping.get(idx, f"Unknown_{idx}")

    def get_cell_type_names(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return [self.get_cell_type_name(i) for i in indices]

    def get_num_cell_types(self):
        return len(self.cell_type_mapping)

    def get_all_cell_types(self):
        return [self.reverse_mapping[i] for i in range(self.get_num_cell_types())]

    def save_cell_type_mapping(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'discovered_types': sorted(self.discovered_types),
                'cell_type_mapping': self.cell_type_mapping,
                'reverse_mapping': {str(k): v for k, v in self.reverse_mapping.items()},
            }, f, indent=2)

    def load_cell_type_mapping(self, filepath):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.discovered_types = set(data['discovered_types'])
        self.cell_type_mapping = data['cell_type_mapping']
        self.reverse_mapping = {int(k): v for k, v in data['reverse_mapping'].items()}