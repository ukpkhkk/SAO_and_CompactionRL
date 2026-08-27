import os
import json
import pandas as pd
import re


DATA_DIR = "./Deepmath-103k"

ALL_OUTPUT = "./deepmath-103k.jsonl"
INTEGER_OUTPUT = "./deepmat-integer.jsonl"


def is_integer_answer(ans):
    """
    判断 final_answer 是否为整数形式

    支持:
        "123"
        "-123"
        "0"
        " 123 "
    不支持:
        "1.5"
        "1/2"
        "\\frac{1}{2}"
    """
    if ans is None:
        return False

    ans = str(ans).strip()

    return re.fullmatch(r"-?\d+", ans) is not None


def main():

    integer_count = 0
    total_count = 0

    with open(ALL_OUTPUT, "w", encoding="utf-8") as fout_all, \
         open(INTEGER_OUTPUT, "w", encoding="utf-8") as fout_int:

        for i in range(10):

            parquet_file = os.path.join(
                DATA_DIR,
                f"train-0000{i}-of-00010.parquet"
            )

            if not os.path.exists(parquet_file):
                raise FileNotFoundError(parquet_file)

            print(f"Reading {parquet_file}")

            df = pd.read_parquet(parquet_file)

            print(f"  samples: {len(df)}")

            for _, row in df.iterrows():

                sample = row.to_dict()

                # 写入完整jsonl
                fout_all.write(
                    json.dumps(
                        sample,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_count += 1

                # 判断整数答案
                if is_integer_answer(sample.get("final_answer")):

                    fout_int.write(
                        json.dumps(
                            sample,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    integer_count += 1


    print("=" * 50)
    print(f"Total samples: {total_count}")
    print(f"Integer final_answer samples: {integer_count}")
    print(f"Saved:")
    print(f"  {ALL_OUTPUT}")
    print(f"  {INTEGER_OUTPUT}")


if __name__ == "__main__":
    main()