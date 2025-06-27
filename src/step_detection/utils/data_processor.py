"""
Data Processing Utilities
Functions for loading and processing step detection data.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_step_data(data_folder: str = "sample_data") -> pd.DataFrame:
    """
    Load and process step detection data from CSV files.

    Args:
        data_folder: Path to the folder containing data files

    Returns:
        Combined DataFrame with processed data
    """
    step_data_frames = []

    print(f"Loading data from: {data_folder}")

    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.endswith(".csv"):
                csv_path = os.path.join(root, filename)
                step_mixed_path = os.path.join(
                    root, filename.replace("Clipped", "") + ".stepMixed"
                )

                if os.path.exists(step_mixed_path):
                    print(f"Processing: {csv_path}")

                    # Read sensor data
                    step_data = pd.read_csv(csv_path, usecols=[1, 2, 3, 4, 5, 6])
                    step_data = step_data.dropna()

                    # Read step indices
                    col_names = ["start_index", "end_index"]
                    step_indices = pd.read_csv(step_mixed_path, names=col_names)
                    step_indices = step_indices.dropna()
                    step_indices = step_indices.loc[
                        (step_indices.end_index < step_data.shape[0])
                    ]

                    # Create labels
                    step_data["Label"] = "No Label"
                    for index, row in step_indices.iterrows():
                        step_data.loc[row["start_index"], "Label"] = "start"
                        step_data.loc[row["end_index"], "Label"] = "end"

                    step_data_frames.append(step_data)

    if not step_data_frames:
        raise ValueError(f"No valid data files found in {data_folder}")

    combined_df = pd.concat(step_data_frames, ignore_index=True)
    print(f"Total samples: {len(combined_df)}")
    print(f"Label distribution:\n{combined_df['Label'].value_counts()}")

    return combined_df


def prepare_data_for_training(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, ...]:
    """
    Prepare data for training.

    Args:
        df: DataFrame with sensor data and labels
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_features, val_features, train_labels, val_labels)
    """
    # Extract features and labels
    features = df.iloc[:, :6].values.astype(np.float32)
    labels = df.iloc[:, 6].values

    # Convert labels to numeric
    label_mapping = {"No Label": 0, "start": 1, "end": 2}
    numeric_labels = np.array([label_mapping[label] for label in labels])

    # Split data
    train_features, val_features, train_labels, val_labels = train_test_split(
        features,
        numeric_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=numeric_labels,
    )

    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")

    return train_features, val_features, train_labels, val_labels


def analyze_data_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of the dataset.

    Args:
        df: DataFrame with sensor data and labels

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "total_samples": len(df),
        "label_counts": df["Label"].value_counts().to_dict(),
        "sensor_stats": {},
    }

    # Analyze sensor data statistics
    sensor_columns = df.columns[:6]
    for col in sensor_columns:
        analysis["sensor_stats"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    return analysis


def save_processed_data(df: pd.DataFrame, filepath: str):
    """Save processed data to file."""
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")


def load_processed_data(filepath: str) -> pd.DataFrame:
    """Load processed data from file."""
    df = pd.read_csv(filepath)
    print(f"Data loaded from: {filepath}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Example usage
    print("Data Processing Utilities")
    print("========================")

    try:
        # Load data
        df = load_step_data()

        # Analyze data
        analysis = analyze_data_distribution(df)
        print(f"\nData Analysis:")
        print(f"Total samples: {analysis['total_samples']}")
        print(f"Label distribution: {analysis['label_counts']}")

        # Prepare for training
        train_X, val_X, train_y, val_y = prepare_data_for_training(df)
        print(f"\nData prepared for training!")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the 'sample_data' folder exists with valid CSV files")
