# configs/config_sequential_fixed.py
"""
修正版 SequentialDRN 配置。

关键约定：
  - ID 0 只表示 <PAD>；
  - 所有真实关系从 ID 1 开始；
  - fixed 训练、评估、规则提取都必须使用这里的关系表。
"""
from pathlib import Path

from configs.config_advanced import ADVANCED_TRAIN_CONFIG, AUGMENTATION_CONFIG


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "data_f70b574f"
OUTPUT_DIR = PROJECT_ROOT / "outputs_advanced_fixed"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_DATA = DATA_DIR / "1.2,1.3,1.4_train_text.csv"
TEST_DATA = DATA_DIR / "1.10_test_text.csv"

RELATIONS = [
    "<PAD>",
    "child",
    "child_in_law",
    "parent",
    "parent_in_law",
    "sibling",
    "sibling_in_law",
    "grandparent",
    "grandchild",
    "nephew",
    "uncle",
    "so",
]

RELATION_TO_ID = {rel: i for i, rel in enumerate(RELATIONS)}
ID_TO_RELATION = {i: rel for i, rel in enumerate(RELATIONS)}
PAD_ID = RELATION_TO_ID["<PAD>"]
VALID_RELATION_IDS = [i for i, rel in enumerate(RELATIONS) if rel != "<PAD>"]

SEQUENTIAL_FIXED_CONFIG = {
    "num_relations": len(RELATIONS),
    "embed_dim": 128,
    "num_templates": 150,
    "num_layers": 4,
    "num_heads": 8,
    "dropout": 0.1,
}

EXTRACTION_FIXED_CONFIG = {
    "min_pattern_sim": 0.6,
    "min_conclusion_sim": 0.6,
}
