import ast
import numpy as np
import os
import pandas as pd
import random

from typing import List, Dict, Any, Union


def clean_system(x: Any) -> str:
    """
    Modifies list in *.csv file to string
    """
    try:
        parsed_list = ast.literal_eval(x)
        if isinstance(parsed_list, list) and len(parsed_list) > 0:
            return str(parsed_list[0])
    except (ValueError, SyntaxError):
        pass

    return x


def create_pairs(
    df_partition: pd.DataFrame, num_neg_per_pos: int = 1
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Generates positive and negative gene pairs within a single partition_id.

    Args:
        df_partition: DataFrame containing genes for one partition_id.
        num_neg_per_pos: Ratio of negative to positive pairs.

    Returns:
        A list of dictionaries containing pairwise features and targets.
    """
    df_partition = df_partition.sort_values("start").reset_index(drop=True)
    n_genes: int = len(df_partition)

    if n_genes < 2:
        return []

    pairs: List[Dict[str, Union[str, int, float]]] = []
    systems: np.ndarray = df_partition["system"].values
    starts: np.ndarray = df_partition["start"].values
    ends: np.ndarray = df_partition["end"].values
    cogs: np.ndarray = df_partition["cog_id"].values
    phyla: np.ndarray = df_partition["phylum"].values
    classes: np.ndarray = df_partition["class"].values

    positives: List[tuple[int, int]] = []
    negatives: List[tuple[int, int]] = []

    # Searching pos and neg pairs
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if systems[i] == systems[j]:
                positives.append((i, j))
            else:
                negatives.append((i, j))

    # Balancing
    num_neg_needed: int = len(positives) * num_neg_per_pos
    if len(negatives) > num_neg_needed and num_neg_needed > 0:
        negatives = random.sample(negatives, num_neg_needed)

    selected_pairs = [(pos, 1) for pos in positives] + [(neg, 0) for neg in negatives]

    # Features
    for (i, j), label in selected_pairs:
        distance: int = int(starts[j] - ends[i])
        genes_between: int = j - i - 1

        len_i: int = ends[i] - starts[i]
        len_j: int = ends[j] - starts[j]
        lengs_diff: int = abs(len_i - len_j)

        pairs.append(
            {
                "cog_1": str(cogs[i]),
                "cog_2": str(cogs[j]),
                "is_same_cog": int(cogs[i] == cogs[j]),
                "distance": distance,
                "genes_between": genes_between,
                "length_diff": lengs_diff,
                "len_1": int(len_i),
                "len_2": int(len_j),
                "is_neighbor": int(distance < 100),
                "phylum": str(phyla[i]),
                "class": str(classes[i]),
                "target": label,
            }
        )

    return pairs


def generate_pairwise_dataset(
    data_path: str, output_path: str, sample_frac: float = 1.0
) -> None:
    """
    Main pipeline for generating the dataset for the model.
    """
    print("---Loading raw data")
    df: pd.DataFrame = pd.read_csv(data_path)

    df["cog_id"].fillna("Unknown", inplace=True)
    df["phylum"].fillna("Unknown", inplace=True)
    df["class"].fillna("Unknown", inplace=True)

    if sample_frac < 1.0:
        unique_partitions: np.ndarray = df["partition_id"].unique()
        sample_size: int = int(len(unique_partitions) * sample_frac)
        sampled_partitions: np.ndarray = np.random.choice(
            unique_partitions, size=sample_size, replace=False
        )
        df = df[df["partition_id"].isin(sampled_partitions)]

    df["system"] = df["system"].apply(clean_system)
    df.dropna(subset=["system"], inplace=True)

    print("---Generating pairs per partition and writing to disk...")
    grouped = df.groupby("partition_id")
    count: int = 0
    total: int = len(grouped)

    chunk_size = 50000
    buffer: List[Dict[str, Union[str, int, float]]] = []

    header_written = False

    for name, group in grouped:
        group_pairs = create_pairs(group, 1)
        buffer.extend(group_pairs)
        count += 1

        if len(buffer) >= chunk_size:
            chunk_df = pd.DataFrame(buffer)
            chunk_df.to_csv(
                output_path, mode="a", header=not header_written, index=False
            )
            header_written = True
            buffer.clear()

        if count % 1000 == 0:
            print(f"-----Processed {count}/{total} partitions")

    if len(buffer) > 0:
        chunk_df = pd.DataFrame(buffer)
        chunk_df.to_csv(output_path, mode="a", header=not header_written, index=False)
        buffer.clear()

    print(f"---Dataset successfully generated and saved to {output_path}")


if __name__ == "__main__":
    out_file = "data/processed/pairwise_cogs.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        os.remove(out_file)

    generate_pairwise_dataset("data/raw/cogs.csv", out_file, sample_frac=1.0)

    df_result = pd.read_csv(out_file)
    print(f"Total dataset shape: {df_result.shape}")

    if not df_result.empty:
        print("---10 samples:")
        print(df_result.sample(10).reset_index(drop=True))
