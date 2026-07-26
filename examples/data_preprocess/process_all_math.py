import argparse
import json
import os
import re
import pandas as pd
import datasets

def remove_boxed(s):
    """Remove a surrounding \\boxed{} wrapper when present."""
    if "\\boxed{" in s:
        left = s.find("\\boxed{")
        right = s.rfind("}")
        return s[left + 7 : right]
    return s

def extract_solution_gsm8k(solution_str):
    """Extract the numeric answer after the GSM8K #### marker."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return remove_boxed(solution_str) # Fallback
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_solution_math(solution_str):
    """Normalize a MATH-style answer string."""
    return remove_boxed(solution_str)

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
BOXED_REWARD_DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"

def format_example(question, ground_truth, subset, split='test', idx=0, **extra_fields):
    return {
        # Keep the reward data_source compatible with the boxed math verifier.
        # The original benchmark name is preserved in extra_info.subset.
        "data_source": BOXED_REWARD_DATA_SOURCE,
        "prompt": [{"role": "user", "content": question + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(ground_truth)
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "subset": subset,
            "orig_question": question,
            **extra_fields,
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math500")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    all_data = []
    global_idx = 0

    print(f"Start processing datasets...")

    try:
        print("Processing GSM8K (test)...")
        ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
        for row in ds:
            gt = extract_solution_gsm8k(row['answer'])
            all_data.append(format_example(
                row['question'],
                gt,
                "gsm8k",
                idx=global_idx,
                source_dataset="openai/gsm8k",
                source_split="test",
                raw_answer=row['answer'],
            ))
            global_idx += 1
    except Exception as e:
        print(f"Error processing GSM8K: {e}")

    try:
        print("Processing MATH-500...")
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        for row in ds:
            gt = row.get('answer') or extract_solution_math(row['solution'])
            all_data.append(format_example(
                row['problem'],
                gt,
                "math500",
                idx=global_idx,
                source_dataset="HuggingFaceH4/MATH-500",
                source_split="test",
                subject=row.get('subject'),
                level=row.get('level'),
                unique_id=row.get('unique_id'),
            ))
            global_idx += 1
    except Exception as e:
        print(f"Error processing MATH-500: {e}")

    try:
        print("Processing AIME 2024...")
        ds = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
        for row in ds:
            all_data.append(format_example(
                row['problem'],
                row['answer'],
                "aime24",
                idx=global_idx,
                source_dataset="HuggingFaceH4/aime_2024",
                source_split="train",
                source_id=row.get('id'),
                url=row.get('url'),
                year=row.get('year'),
            ))
            global_idx += 1
    except Exception as e:
        print(f"Error processing AIME 2024: {e}")

    try:
        print("Processing AIME 2025...")
        ds = datasets.load_dataset("MathArena/aime_2025", split="train")
        for row in ds:
            all_data.append(format_example(
                row['problem'],
                row['answer'],
                "aime25",
                idx=global_idx,
                source_dataset="MathArena/aime_2025",
                source_split="train",
                source_id=row.get('problem_idx'),
                problem_type=row.get('problem_type'),
            ))
            global_idx += 1
    except Exception as e:
        print(f"Error processing AIME 2025: {e}")

    try:
        print("Processing AMC 2023...")
        ds = datasets.load_dataset("knoveleng/AMC-23", split="train")
        for row in ds:
            all_data.append(format_example(
                row.get('problem') or row.get('question'),
                row['answer'],
                "amc23",
                idx=global_idx,
                source_dataset="knoveleng/AMC-23",
                source_split="train",
                source_id=row.get('id'),
                url=row.get('url'),
            ))
            global_idx += 1

    except Exception as e:
        print(f"Error processing AMC 2023: {e}")

    print(f"\nTotal samples collected: {len(all_data)}")
    
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        
        print("Reward data source distribution:")
        print(df['data_source'].value_counts())
        print("Subset distribution:")
        print(df['extra_info'].map(lambda x: x['subset']).value_counts())

        local_dir = os.path.expanduser(args.local_dir)
        os.makedirs(local_dir, exist_ok=True)

        subsets = df['extra_info'].map(lambda x: x['subset'])
        output_path = os.path.join(local_dir, 'test.parquet')
        df.to_parquet(output_path)
        print(f"\nSaved combined test dataset to: {output_path}")

        aime_df = df[subsets.isin(["aime24", "aime25"])].reset_index(drop=True)
        general_df = df[subsets.isin(["gsm8k", "math500", "amc23"])].reset_index(drop=True)
        aime_output_path = os.path.join(local_dir, "test_aime.parquet")
        general_output_path = os.path.join(local_dir, "test_general.parquet")
        aime_df.to_parquet(aime_output_path)
        general_df.to_parquet(general_output_path)
        print(f"Saved AIME-only test dataset to: {aime_output_path}")
        print(f"Saved general test dataset to: {general_output_path}")

        manifest = {
            "reward_data_source": BOXED_REWARD_DATA_SOURCE,
            "instruction": INSTRUCTION,
            "num_rows": len(df),
            "subset_counts": subsets.value_counts().to_dict(),
            "outputs": {
                "combined": {
                    "path": output_path,
                    "num_rows": len(df),
                    "subsets": subsets.value_counts().to_dict(),
                },
                "aime": {
                    "path": aime_output_path,
                    "num_rows": len(aime_df),
                    "subsets": aime_df['extra_info'].map(lambda x: x['subset']).value_counts().to_dict(),
                    "recommended_use": "long-response AIME evaluation",
                },
                "general": {
                    "path": general_output_path,
                    "num_rows": len(general_df),
                    "subsets": general_df['extra_info'].map(lambda x: x['subset']).value_counts().to_dict(),
                    "recommended_use": "standard-budget GSM8K, MATH-500, and AMC-23 evaluation",
                },
            },
        }
        manifest_path = os.path.join(local_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Saved manifest to: {manifest_path}")

        if args.hdfs_dir is not None:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
            print(f"Uploaded to HDFS: {args.hdfs_dir}")
    else:
        print("No data collected!")
