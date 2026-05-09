# utils/data_loader_fixed.py
"""
修正版 CLUTRR 数据加载器。

这个版本使用 config_sequential_fixed.py 里的关系 ID：
  - <PAD> = 0；
  - child 等真实关系从 1 开始。
"""
import ast
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config_sequential_fixed import PAD_ID, RELATION_TO_ID


class FixedCLUTRRDataset(Dataset):
    """使用独立 PAD ID 的 CLUTRR 数据集。"""

    def __init__(self, csv_path: str, max_chain_length: int = 10):
        self.csv_path = csv_path
        self.max_chain_length = max_chain_length
        self.samples = []
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        for idx, row in df.iterrows():
            try:
                sample = self._parse_row(row)
                if sample is not None:
                    self.samples.append(sample)
            except Exception as exc:
                if idx < 5:
                    print(f"Warning: Failed to parse row {idx}: {exc}")

        print(f"Loaded {len(self.samples)} valid samples")
        self._print_statistics()

    def _parse_row(self, row):
        if "edge_types" not in row:
            return None

        edge_types = row["edge_types"]
        if isinstance(edge_types, str):
            edge_types = ast.literal_eval(edge_types)

        relation_chain = []
        for rel in edge_types:
            rel = self._normalize_relation(rel)
            if rel in RELATION_TO_ID:
                relation_chain.append(RELATION_TO_ID[rel])

        if not relation_chain:
            return None

        chain_length = min(len(relation_chain), self.max_chain_length)
        relation_chain = relation_chain[: self.max_chain_length]
        relation_chain += [PAD_ID] * (self.max_chain_length - len(relation_chain))

        target = row.get("target", None)
        if not target:
            return None

        target = self._normalize_relation(target)
        if target not in RELATION_TO_ID or target == "<PAD>":
            return None

        return {
            "chain": relation_chain,
            "chain_length": chain_length,
            "target": RELATION_TO_ID[target],
            "target_name": target,
        }

    def _normalize_relation(self, rel: str) -> str:
        rel = str(rel).lower().strip()
        mapping = {
            "father": "parent",
            "mother": "parent",
            "son": "child",
            "daughter": "child",
            "brother": "sibling",
            "sister": "sibling",
            "grandfather": "grandparent",
            "grandmother": "grandparent",
            "grandson": "grandchild",
            "granddaughter": "grandchild",
            "husband": "so",
            "wife": "so",
            "aunt": "uncle",
            "niece": "nephew",
            "neice": "nephew",
            "son-in-law": "child_in_law",
            "daughter-in-law": "child_in_law",
            "father-in-law": "parent_in_law",
            "mother-in-law": "parent_in_law",
        }
        return mapping.get(rel, rel)

    def _print_statistics(self):
        target_counts = Counter(sample["target_name"] for sample in self.samples)
        print("Dataset statistics:")
        print(f"  Total samples: {len(self.samples)}")
        print("  Target distribution:")
        for target, count in sorted(target_counts.items(), key=lambda item: -item[1])[:10]:
            print(f"    {target}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "chain": torch.LongTensor(sample["chain"]),
            "chain_length": sample["chain_length"],
            "target": torch.LongTensor([sample["target"]]),
        }


def get_fixed_data_loaders(train_path, test_path, batch_size=32, max_chain_length=10):
    train_dataset = FixedCLUTRRDataset(train_path, max_chain_length)
    test_dataset = FixedCLUTRRDataset(test_path, max_chain_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader
