import pandas as pd
import argparse
import os
import json

def main(report_path, output_path, metric):
    # Read the report
    df = pd.read_csv(report_path)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in the report. Available columns: {df.columns.tolist()}")

    # Find the row with the best model (highest metric)
    best_row = df.loc[df[metric].idxmax()]
    best_model = best_row['Model']

    # Save selected model to a JSON file
    result = {
        "selected_model": best_model,
        "metric": metric,
        "score": best_row[metric]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✅ Selected model: {best_model} (based on {metric}: {best_row[metric]:.4f})")
    print(f"📄 Saved selection to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the best model from cross-validation report.")
    parser.add_argument("--report", type=str, required=True, help="Path to the cross-validation report CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON with selected model")
    parser.add_argument("--metric", type=str, default="F1 Mean", help="Metric column to select best model")

    args = parser.parse_args()
    main(args.report, args.output, args.metric)