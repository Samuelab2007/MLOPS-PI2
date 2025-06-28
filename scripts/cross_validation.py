import os
import sys
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


def main(data_path: str, report_path: str, verbose: bool = False):

    try:
        # PATHS
        SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        DATA_DIR = os.path.join(SCRIPT_DIR, data_path)
        REPORT_PATH = os.path.join(SCRIPT_DIR, report_path)

        df = pd.read_csv(DATA_DIR)

        if verbose:
            print(f"Loading dataset from {data_path}")

        # Split the dataset into features and target
        imputer = SimpleImputer(strategy='mean')
        X = df.drop('target', axis=1)
        X = imputer.fit_transform(X)
        y = df['target']

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)


        # Define the ML models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=0),
            'Random Forest': RandomForestClassifier(random_state=0),
            'Gradient Boosting': GradientBoostingClassifier(random_state=0),
            'Support Vector Machine': SVC(random_state=0),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='binary'),
            'recall': make_scorer(recall_score, average='binary'),
            'f1': make_scorer(f1_score, average='binary')
        }

        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=5)

        # Evaluate each model
        report_rows = []

        for name, model in models.items():
            results = cross_validate(model, X, y, cv=cv, scoring=scoring)
            row = {
                "Model": name,
                "Accuracy Mean": results["test_accuracy"].mean(),
                "Accuracy Std": results["test_accuracy"].std(),
                "Precision Mean": results["test_precision"].mean(),
                "Precision Std": results["test_precision"].std(),
                "Recall Mean": results["test_recall"].mean(),
                "Recall Std": results["test_recall"].std(),
                "F1 Mean": results["test_f1"].mean(),
                "F1 Std": results["test_f1"].std()
            }

            report_rows.append(row)
            if verbose:
                print(f"\nEvaluating {name}...")
                for metric in scoring.keys():
                    scores = results[f'test_{metric}']
                    print(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")

        # Save report to CSV
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(REPORT_PATH, index=False)

        if verbose:
            print(f"Saved report to {report_path}")

    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation reporting script.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV input dataset")
    parser.add_argument("--report", type=str, required=True, help="Path to CSV output report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    main(args.data, args.report, args.verbose)