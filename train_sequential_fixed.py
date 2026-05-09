# scripts/train_sequential_fixed.py
"""
训练修正版 SequentialDRN。

这个版本不修改原来的 train_sequential.py，主要修复：
  1. <PAD> 单独占用 ID 0，真实关系从 ID 1 开始；
  2. 数据加载、模型类别数、评估标签使用同一套关系 ID；
  3. 不再把 child 当作 padding；
  4. 训练输出保存到独立目录，避免覆盖旧实验结果。

可选：
  在训练前用大模型从 story/query 文本抽取三元组，再通过符号验证器生成
  edge_types，供 SequentialDRN 训练使用。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config_sequential_fixed import (  # noqa: E402
    ADVANCED_TRAIN_CONFIG,
    AUGMENTATION_CONFIG,
    ID_TO_RELATION,
    OUTPUT_DIR,
    RELATION_TO_ID,
    RELATIONS,
    SEQUENTIAL_FIXED_CONFIG,
    TEST_DATA,
    TRAIN_DATA,
)
from models.sequential_drn import SequentialDRN  # noqa: E402
from scripts.extract_llm_triples import (  # noqa: E402
    DEFAULT_API_URL,
    build_messages,
    call_llm,
    load_done_ids,
    read_rows,
    row_id,
    triples_from_llm_json,
    write_jsonl_item,
)
from scripts.preprocess_verifier_guided_chains import process as preprocess_text_chains  # noqa: E402
from utils.curriculum_trainer import CurriculumTrainer  # noqa: E402
from utils.data_augmentation import CLUTRRAugmenter  # noqa: E402
from utils.data_loader_fixed import get_fixed_data_loaders  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train fixed SequentialDRN")
    parser.add_argument("--train-data", default=str(TRAIN_DATA), help="training CSV path")
    parser.add_argument("--test-data", default=str(TEST_DATA), help="test/validation CSV path")
    parser.add_argument(
        "--use-llm-triples",
        action="store_true",
        help="extract triples from text with an LLM before training",
    )
    parser.add_argument("--train-vg", default=None, help="output CSV after train text-to-chain preprocessing")
    parser.add_argument("--test-vg", default=None, help="output CSV after test text-to-chain preprocessing")
    parser.add_argument("--llm-train-jsonl", default=None, help="LLM triple JSONL for train rows")
    parser.add_argument("--llm-test-jsonl", default=None, help="LLM triple JSONL for test rows")
    parser.add_argument(
        "--skip-llm-extract",
        action="store_true",
        help="reuse existing --llm-train-jsonl/--llm-test-jsonl and only run verifier preprocessing",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="reuse existing --train-vg/--test-vg when --use-llm-triples is enabled",
    )
    parser.add_argument("--no-fallback", action="store_true", help="drop rows when verifier cannot build a chain")
    parser.add_argument("--llm-limit", type=int, default=None, help="extract only first N unfinished rows per split")
    parser.add_argument("--api-url", default=os.environ.get("LLM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"))
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep after each successful LLM request")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-retry-wait", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument("--no-resume-llm", action="store_true", help="overwrite existing LLM JSONL outputs")
    return parser.parse_args()


def default_vg_path(input_path):
    input_path = Path(input_path)
    stem = input_path.stem.replace("_text", "")
    return input_path.with_name(f"{stem}_llm_vg_text.csv")


def default_llm_jsonl_path(input_path):
    input_path = Path(input_path)
    return OUTPUT_DIR / f"llm_triples_{input_path.stem}.jsonl"


def extract_llm_triples_for_csv(input_path, output_path, args):
    rows = read_rows(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume = not args.no_resume_llm
    done_ids = load_done_ids(output_path) if resume else set()
    todo = [row for row in rows if row_id(row) not in done_ids]
    if args.llm_limit is not None:
        todo = todo[: args.llm_limit]

    llm_args = SimpleNamespace(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_format_json=args.response_format_json,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_retry_wait=args.max_retry_wait,
    )

    mode = "a" if resume else "w"
    ok = 0
    failed = 0
    with open(output_path, mode) as f:
        for row in tqdm(todo, desc=f"LLM triples: {Path(input_path).name}"):
            rid = row_id(row)
            try:
                response = call_llm(llm_args, build_messages(row))
                try:
                    triples = triples_from_llm_json(response)
                    parse_error = None
                except Exception as exc:
                    triples = []
                    parse_error = str(exc)

                write_jsonl_item(
                    f,
                    {
                        "id": rid,
                        "response": response,
                        "triples": triples,
                        "parse_error": parse_error,
                    },
                )
                ok += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as exc:
                failed += 1
                write_jsonl_item(
                    f,
                    {
                        "id": rid,
                        "response": "",
                        "triples": [],
                        "error": str(exc),
                    },
                )

    print(f"LLM triples saved: {output_path} (new ok={ok}, failed={failed}, skipped={len(rows) - len(todo)})")


def prepare_llm_text_data(args):
    train_data = Path(args.train_data)
    test_data = Path(args.test_data)
    train_vg = Path(args.train_vg) if args.train_vg else default_vg_path(train_data)
    test_vg = Path(args.test_vg) if args.test_vg else default_vg_path(test_data)
    llm_train_jsonl = Path(args.llm_train_jsonl) if args.llm_train_jsonl else default_llm_jsonl_path(train_data)
    llm_test_jsonl = Path(args.llm_test_jsonl) if args.llm_test_jsonl else default_llm_jsonl_path(test_data)

    if args.no_preprocess:
        return train_vg, test_vg

    if not args.skip_llm_extract:
        if not args.api_key:
            raise RuntimeError("Set --api-key or LLM_API_KEY/OPENROUTER_API_KEY before using --use-llm-triples")
        extract_llm_triples_for_csv(train_data, llm_train_jsonl, args)
        extract_llm_triples_for_csv(test_data, llm_test_jsonl, args)

    preprocess_text_chains(
        input_path=str(train_data),
        output_path=str(train_vg),
        llm_jsonl=str(llm_train_jsonl),
        fallback=not args.no_fallback,
    )
    preprocess_text_chains(
        input_path=str(test_data),
        output_path=str(test_vg),
        llm_jsonl=str(llm_test_jsonl),
        fallback=not args.no_fallback,
    )
    return train_vg, test_vg


def main():
    args = parse_args()
    print("\n" + "=" * 70)
    print("训练修正版序列编码规则网络 (SequentialDRN)")
    print("=" * 70)
    print("修复点: <PAD>=0，真实关系从 1 开始，避免 child 被当作 padding")

    train_data = Path(args.train_data)
    test_data = Path(args.test_data)
    if args.use_llm_triples:
        print("\n启用 LLM 文本->三元组->验证链预处理")
        train_data, test_data = prepare_llm_text_data(args)

    print(f"\n{'=' * 70}")
    print("[1/5] 加载数据")
    print(f"{'=' * 70}")

    train_loader, val_loader = get_fixed_data_loaders(
        str(train_data),
        str(test_data),
        batch_size=ADVANCED_TRAIN_CONFIG["batch_size"],
        max_chain_length=10,
    )

    print(f"\n{'=' * 70}")
    print("[2/5] 数据增强")
    print(f"{'=' * 70}")

    augmenter = CLUTRRAugmenter(**AUGMENTATION_CONFIG)
    train_dataset = augmenter.augment_dataset(train_loader.dataset)

    print(f"\n{'=' * 70}")
    print("[3/5] 创建模型")
    print(f"{'=' * 70}")

    model_config = dict(SEQUENTIAL_FIXED_CONFIG)
    model = SequentialDRN(**model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("模型创建完成")
    print(f"  关系数量: {len(RELATIONS)}，其中 <PAD> 只用于填充")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: ~{total_params * 4 / (1024 * 1024):.1f} MB")

    print(f"\n{'=' * 70}")
    print("[4/5] 课程学习训练")
    print(f"{'=' * 70}")

    trainer = CurriculumTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_loader.dataset,
        config=ADVANCED_TRAIN_CONFIG,
        output_dir=OUTPUT_DIR,
    )
    history = trainer.train()

    print(f"\n{'=' * 70}")
    print("[5/5] 保存模型")
    print(f"{'=' * 70}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "relations": RELATIONS,
            "relation_to_id": RELATION_TO_ID,
            "id_to_relation": ID_TO_RELATION,
            "model_config": model_config,
        },
        OUTPUT_DIR / "sequential_fixed_final.pt",
    )

    serializable_history = {
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
    }
    with open(OUTPUT_DIR / "training_history_sequential_fixed.json", "w") as f:
        json.dump(serializable_history, f, indent=2)

    with open(OUTPUT_DIR / "relation_vocab.json", "w") as f:
        json.dump(
            {
                "relations": RELATIONS,
                "relation_to_id": RELATION_TO_ID,
                "id_to_relation": ID_TO_RELATION,
            },
            f,
            indent=2,
        )

    print(f"\n所有文件已保存到: {OUTPUT_DIR}")
    print("下一步: python3 scripts/evaluate_advanced_fixed.py")
    print("再提取规则: python3 scripts/extract_rules_fixed.py")


if __name__ == "__main__":
    main()
