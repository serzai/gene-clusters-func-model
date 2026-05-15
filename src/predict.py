import pandas as pd
import joblib
import argparse
import os
import warnings
warnings.filterwarnings("ignore")


def build_features(
    cog_1: str,
    start_1: int,
    end_1: int,
    cog_2: str,
    start_2: int,
    end_2: int,
    phylum: str,
    tax_class: str,
) -> pd.DataFrame:
    """
    Computes pairwise features from raw gene coordinates.
    """
    if start_1 > start_2:
        cog_1, cog_2 = cog_2, cog_1
        start_1, start_2 = start_2, start_1
        end_1, end_2 = end_2, end_1

    distance: int = start_2 - end_1
    length_1: int = end_1 - start_1
    length_2: int = end_2 - start_2
    length_diff: int = abs(length_1 - length_2)

    avg_gene_length = 1000
    genes_between: int = max(0, distance // avg_gene_length)

    return pd.DataFrame(
        [
            {
                "cog_1": str(cog_1),
                "cog_2": str(cog_2),
                "distance": distance,
                "genes_between": genes_between,
                "length_diff": length_diff,
                "phylum": str(phylum),
                "class": str(tax_class),
            }
        ]
    )


def predict_pair(
    model_path: str,
    cog_1: str,
    start_1: int,
    end_1: int,
    cog_2: str,
    start_2: int,
    end_2: int,
    phylum: str,
    tax_class: str,
) -> None:
    """
    Loads the trained model and predicts the probability of a functional
    relationship between two genes using their raw genomic coordinates.
    """
    if not os.path.exists(model_path):
        print(
            f"Error: Model not found at {model_path}. Train the model first.")
        return

    pipeline = joblib.load(model_path)

    input_data = build_features(
        cog_1, start_1, end_1, cog_2, start_2, end_2, phylum, tax_class
    )

    distance = int(input_data["distance"].iloc[0])
    genes_between = int(input_data["genes_between"].iloc[0])
    length_diff = int(input_data["length_diff"].iloc[0])

    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    prob_related = probabilities[1] * 100

    print("\n" + "=" * 44)
    print(f"Gene 1:   {cog_1}  [{start_1} → {end_1}]")
    print(f"Gene 2:   {cog_2}  [{start_2} → {end_2}]")
    print(f"Taxonomy: {phylum} / {tax_class}")
    print("-" * 44)
    print(f"Distance:      {distance:>+10,} bp")
    print(f"Genes between: {genes_between:>10}")
    print(f"Length diff:   {length_diff:>10} bp")
    print("-" * 44)

    if prediction == 1:
        print("Verdict: ✅ RELATED")
    else:
        print("Verdict: ❌ NOT RELATED")

    print(f"Confidence: {prob_related:.2f}%")
    print("=" * 44 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict functional relationship between two genes using raw genomic coordinates.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model_pipeline.joblib",
        help="Path to the trained joblib model.",
    )
    parser.add_argument(
        "--cog1", type=str, required=True, help="ID of gene 1"
    )
    parser.add_argument(
        "--start1", type=int, required=True, help="Start coordinate of gene 1"
    )
    parser.add_argument(
        "--end1", type=int, required=True, help="End coordinate of gene 1"
    )
    parser.add_argument(
        "--cog2", type=str, required=True, help="ID of gene 2"
    )
    parser.add_argument(
        "--start2", type=int, required=True, help="Start coordinate of gene 2"
    )
    parser.add_argument(
        "--end2", type=int, required=True, help="End coordinate of gene 2"
    )
    parser.add_argument(
        "--phylum", type=str, default="Unknown", help="Phylum of the organism"
    )
    parser.add_argument(
        "--tax_class",
        type=str,
        default="Unknown",
        help="Taxonomic class of the organism",
    )

    args = parser.parse_args()

    predict_pair(
        model_path=args.model,
        cog_1=args.cog1,
        start_1=args.start1,
        end_1=args.end1,
        cog_2=args.cog2,
        start_2=args.start2,
        end_2=args.end2,
        phylum=args.phylum,
        tax_class=args.tax_class,
    )
