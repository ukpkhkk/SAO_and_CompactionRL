import json
import random


INPUT_FILE = "deepmath-40k.jsonl"
OUTPUT_FILE = "deepmath-17k.jsonl"

SAMPLE_NUM = 17000
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    # 读取全部样本
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = f.readlines()

    total = len(data)

    if total < SAMPLE_NUM:
        raise ValueError(
            f"Dataset only has {total} samples, "
            f"cannot sample {SAMPLE_NUM}"
        )

    print(f"Total samples: {total}")

    # 随机选择索引
    sampled_lines = random.sample(data, SAMPLE_NUM)

    # 写入新的 jsonl
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in sampled_lines:
            f.write(line)

    print("=" * 50)
    print(f"Sampled samples: {SAMPLE_NUM}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()