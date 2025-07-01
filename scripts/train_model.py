import pandas as pd
import json
import argparse
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Define model templates and parameter grids
MODEL_CONFIGS = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=0),
        "params": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=0),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=0),
        "params": {
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [100, 200]
        }
    },
    "Support Vector Machine": {
        "model": SVC(random_state=0),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    }
}

def load_selected_model(path):
    with open(path, 'r') as f:
        return json.load(f)["selected_model"]

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    return X, y

def main(data_path, selection_path, model_output_path):
    # Load selected model
    selected_model_name = load_selected_model(selection_path)
    print(f"🔍 Selected model: {selected_model_name}")

    # Load model and param grid
    model_info = MODEL_CONFIGS.get(selected_model_name)
    if model_info is None:
        raise ValueError(f"No configuration found for model '{selected_model_name}'")

    model = model_info["model"]
    param_grid = model_info["params"]

    # Load and preprocess data
    X, y = preprocess_data(data_path)

    # Run GridSearchCV
    print("🔧 Tuning hyperparameters...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    print(f"✅ Best hyperparameters: {grid_search.best_params_}")

    # Save final model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"💾 Trained model saved to {model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain selected model with hyperparameter tuning.")
    parser.add_argument("--data", required=True, help="Path to preprocessed dataset CSV")
    parser.add_argument("--selection", required=True, help="Path to selected_model.json")
    parser.add_argument("--output", required=True, help="Path to save final model (e.g. .joblib)")
    args = parser.parse_args()

    main(args.data, args.selection, args.output)