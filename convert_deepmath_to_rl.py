import json


INPUT_FILE = "deepmath-17k.jsonl"
OUTPUT_FILE = "deepmath-17k-rl.jsonl"


PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{question}

Remember to put your answer on its own line after "Answer:"."""


def main():

    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:

            if not line.strip():
                continue

            sample = json.loads(line)

            question = sample["question"]
            answer = sample["final_answer"]

            # 构造prompt
            prompt = PROMPT_TEMPLATE.format(
                question=question
            )

            # RL训练格式
            rl_sample = {
                "prompt": [
                    {
                        "content": prompt,
                        "role": "user"
                    }
                ],
                "label": str(answer)
            }

            fout.write(
                json.dumps(
                    rl_sample,
                    ensure_ascii=False
                ) + "\n"
            )

            count += 1


    print("=" * 50)
    print(f"Converted samples: {count}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()