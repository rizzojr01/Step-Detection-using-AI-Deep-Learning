#!/usr/bin/env python3
"""
Comprehensive dataset analysis to identify issues affecting model performance.
This script will help us understand why the model predicts 99%+ 'No Step'.
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from step_detection.utils.data_processor import load_step_data
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def analyze_dataset_structure(df):
    """First, let's understand the actual structure of our dataset."""
    print("🔍 DATASET STRUCTURE ANALYSIS")
    print("=" * 50)

    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")

    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())

    # Check for numeric columns (likely sensor data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns: {numeric_cols}")

    return numeric_cols


def analyze_label_distribution(df):
    """Analyze the distribution of labels in the dataset."""
    print("\n📊 LABEL DISTRIBUTION ANALYSIS")
    print("=" * 50)

    label_counts = df["Label"].value_counts()
    total_samples = len(df)

    print(f"Total samples: {total_samples:,}")
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {label:<12}: {count:>8,} ({percentage:>6.2f}%)")

    # Calculate class imbalance ratio
    no_label_count = label_counts.get("No Label", 0)
    start_count = label_counts.get("start", 0)
    end_count = label_counts.get("end", 0)

    if start_count > 0 and end_count > 0:
        imbalance_ratio = no_label_count / (start_count + end_count)
        print(
            f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1 (No Label : Step Labels)"
        )

        if imbalance_ratio > 50:
            print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            print(
                "   This extreme imbalance is likely causing the model to predict mostly 'No Step'"
            )
        elif imbalance_ratio > 20:
            print("⚠️  SIGNIFICANT CLASS IMBALANCE DETECTED!")
            print("   This imbalance is contributing to poor model performance")

    return label_counts


def analyze_sensor_data_quality(df, numeric_cols):
    """Analyze the quality and characteristics of sensor data."""
    print("\n🔍 SENSOR DATA QUALITY ANALYSIS")
    print("=" * 50)

    if not numeric_cols:
        print("❌ No numeric columns found for sensor data analysis")
        return None

    # Remove 'Label' column if it's in numeric_cols
    sensor_columns = [col for col in numeric_cols if col != "Label"]

    if len(sensor_columns) < 6:
        print(
            f"⚠️  Expected 6 sensor columns, found {len(sensor_columns)}: {sensor_columns}"
        )

    print(f"Analyzing sensor columns: {sensor_columns}")

    # Check for missing values
    print("\nMissing values:")
    missing_data = df[sensor_columns].isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")

    if missing_data.sum() == 0:
        print("  ✅ No missing values found")

    # Check for infinite or extreme values
    print("\nExtreme values check:")
    for col in sensor_columns:
        col_data = df[col]
        inf_count = np.isinf(col_data).sum()
        extreme_count = (np.abs(col_data) > 1000).sum()

        if inf_count > 0 or extreme_count > 0:
            print(f"  {col}: {inf_count} infinite, {extreme_count} extreme (>1000)")
        else:
            print(f"  {col}: ✅ No extreme values")

    # Basic statistics
    print("\nSensor data statistics:")
    stats = df[sensor_columns].describe()
    print(stats.round(3))

    return stats


def check_data_consistency(df, numeric_cols):
    """Check for data consistency issues."""
    print("\n🔍 DATA CONSISTENCY CHECK")
    print("=" * 50)

    issues_found = []
    sensor_columns = [col for col in numeric_cols if col != "Label"]

    if len(sensor_columns) >= 3:
        # Assume first 3 are accelerometer, next 3 are gyroscope
        accel_cols = sensor_columns[:3]
        gyro_cols = sensor_columns[3:6] if len(sensor_columns) >= 6 else []

        print(f"Assumed accelerometer columns: {accel_cols}")
        print(f"Assumed gyroscope columns: {gyro_cols}")

        # Check accelerometer magnitude
        if len(accel_cols) == 3:
            accel_magnitude = np.sqrt(
                df[accel_cols[0]] ** 2 + df[accel_cols[1]] ** 2 + df[accel_cols[2]] ** 2
            )
            mean_accel_mag = accel_magnitude.mean()

            print(f"\nMean accelerometer magnitude: {mean_accel_mag:.2f}")
            if mean_accel_mag < 5 or mean_accel_mag > 15:
                issues_found.append(
                    "Unusual accelerometer magnitude - check units/calibration"
                )
                print("  ⚠️  Unusual accelerometer magnitude - check units/calibration")
            else:
                print("  ✅ Accelerometer magnitude looks reasonable")

        # Check gyroscope magnitude
        if len(gyro_cols) == 3:
            gyro_magnitude = np.sqrt(
                df[gyro_cols[0]] ** 2 + df[gyro_cols[1]] ** 2 + df[gyro_cols[2]] ** 2
            )
            mean_gyro_mag = gyro_magnitude.mean()

            print(f"Mean gyroscope magnitude: {mean_gyro_mag:.2f}")
            if mean_gyro_mag > 10:
                issues_found.append("High gyroscope values - check units")
                print("  ⚠️  High gyroscope values - check units")
            else:
                print("  ✅ Gyroscope magnitude looks reasonable")

    # Check for constant values (sensor stuck)
    print(f"\nChecking for constant/stuck sensors:")
    for col in sensor_columns:
        unique_values = df[col].nunique()
        if unique_values < 10:
            issues_found.append(f"{col} has very few unique values ({unique_values})")
            print(
                f"  ⚠️  {col} has very few unique values ({unique_values}) - sensor might be stuck"
            )
        else:
            print(f"  ✅ {col} has {unique_values} unique values")

    return issues_found


def analyze_step_patterns(df):
    """Analyze patterns in step start/end labels."""
    print("\n👣 STEP PATTERN ANALYSIS")
    print("=" * 50)

    # Find step sequences
    step_starts = df[df["Label"] == "start"].index.tolist()
    step_ends = df[df["Label"] == "end"].index.tolist()

    print(f"Step starts found: {len(step_starts)}")
    print(f"Step ends found: {len(step_ends)}")

    if len(step_starts) != len(step_ends):
        print("⚠️  MISMATCH: Number of step starts and ends don't match!")
        print("   This could indicate labeling issues")
    else:
        print("✅ Step starts and ends are balanced")

    # Analyze step durations (if starts and ends are paired)
    if len(step_starts) > 0 and len(step_ends) > 0:
        step_durations = []
        valid_steps = 0

        for start_idx in step_starts:
            # Find the next end after this start
            next_ends = [end for end in step_ends if end > start_idx]
            if next_ends:
                end_idx = min(next_ends)
                duration = end_idx - start_idx
                if duration > 0 and duration < 1000:  # Reasonable step duration
                    step_durations.append(duration)
                    valid_steps += 1

        if step_durations:
            print(f"\nStep duration analysis (valid steps: {valid_steps}):")
            print(f"  Mean duration: {np.mean(step_durations):.1f} samples")
            print(f"  Median duration: {np.median(step_durations):.1f} samples")
            print(f"  Min duration: {np.min(step_durations)} samples")
            print(f"  Max duration: {np.max(step_durations)} samples")

            # Check for very short or very long steps
            short_steps = sum(1 for d in step_durations if d < 5)
            long_steps = sum(1 for d in step_durations if d > 100)

            if short_steps > 0:
                print(
                    f"  ⚠️  {short_steps} very short steps (<5 samples) - possible noise"
                )
            if long_steps > 0:
                print(
                    f"  ⚠️  {long_steps} very long steps (>100 samples) - possible errors"
                )

            if short_steps == 0 and long_steps == 0:
                print("  ✅ All step durations look reasonable")


def generate_recommendations(label_counts, issues_found):
    """Generate recommendations based on analysis results."""
    print("\n💡 RECOMMENDATIONS TO FIX MODEL PERFORMANCE")
    print("=" * 60)

    recommendations = []

    # Check class imbalance
    no_label_count = label_counts.get("No Label", 0)
    start_count = label_counts.get("start", 0)
    end_count = label_counts.get("end", 0)

    if no_label_count > 0 and (start_count + end_count) > 0:
        imbalance_ratio = no_label_count / (start_count + end_count)

        print(f"🎯 PRIMARY ISSUE: Class Imbalance Ratio = {imbalance_ratio:.1f}:1")

        if imbalance_ratio > 20:
            print(
                "\n🚨 CRITICAL: This imbalance is causing 99%+ 'No Step' predictions!"
            )
            print("\n🔧 IMMEDIATE SOLUTIONS:")
            print("1. 📊 Use Strong Class Weights:")
            print("   class_weight = {0: 1.0, 1: 25.0, 2: 25.0}")
            print("   # This gives 25x more importance to step classes")

            print("\n2. 🎲 Data Augmentation for Step Classes:")
            print("   - Add noise to step samples")
            print("   - Time shifting")
            print("   - Scaling variations")

            print("\n3. 🎯 Threshold Optimization:")
            print("   - Lower confidence threshold from 0.8 to 0.5")
            print("   - Use focal loss to focus on hard examples")
            
            print("\n4. 🔄 Balanced Sampling:")
            print("   - Undersample 'No Label' class")
            print("   - Oversample step classes")
            print("   - Use SMOTE for synthetic step samples")
            
            recommendations.extend([
                "Apply strong class weights (25:1 ratio)",
                "Use data augmentation for step classes",
                "Lower prediction thresholds",
                "Consider balanced sampling strategies"
            ])
        
        elif imbalance_ratio > 10:
            print("\n⚠️  MODERATE: Imbalance is affecting performance")
            print("\n🔧 SOLUTIONS:")
            print("1. 📊 Use Moderate Class Weights:")
            print(f"   class_weight = {{0: 1.0, 1: {imbalance_ratio/2:.1f}, 2: {imbalance_ratio/2:.1f}}}")
            
            print("\n2. 🎯 Adjust Thresholds:")
            print("   - Lower confidence threshold to 0.6")
            print("   - Use precision-recall optimization")
            
            recommendations.extend([
                "Apply moderate class weights",
                "Adjust prediction thresholds",
                "Monitor precision-recall balance"
            ])
    
    # Check for data quality issues
    if issues_found:
        print(f"\n🔍 DATA QUALITY ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")
            recommendations.append(f"Fix: {issue}")
    
    # General recommendations
    print(f"\n📋 IMPLEMENTATION CHECKLIST:")
    print("□ Retrain model with class weights")
    print("□ Implement data augmentation")
    print("□ Optimize detection thresholds")
    print("□ Add validation with real-world testing")
    print("□ Monitor false positive/negative rates")
    
    return recommendations


def create_visualization(df, numeric_cols):
    """Create visualizations to understand the data better."""
    print("\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Analysis Visualizations', fontsize=16)
        
        # 1. Label distribution
        label_counts = df["Label"].value_counts()
        axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Label Distribution')
        
        # 2. Sensor data distribution (first sensor column)
        sensor_columns = [col for col in numeric_cols if col != "Label"]
        if sensor_columns:
            first_sensor = sensor_columns[0]
            axes[0, 1].hist(df[first_sensor], bins=50, alpha=0.7)
            axes[0, 1].set_title(f'{first_sensor} Distribution')
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Step vs No Step comparison (if we have enough sensor columns)
        if len(sensor_columns) >= 2:
            step_data = df[df["Label"].isin(["start", "end"])]
            no_step_data = df[df["Label"] == "No Label"].sample(min(len(step_data), 1000))
            
            axes[1, 0].scatter(step_data[sensor_columns[0]], step_data[sensor_columns[1]], 
                             alpha=0.5, label='Steps', s=1)
            axes[1, 0].scatter(no_step_data[sensor_columns[0]], no_step_data[sensor_columns[1]], 
                             alpha=0.5, label='No Steps', s=1)
            axes[1, 0].set_xlabel(sensor_columns[0])
            axes[1, 0].set_ylabel(sensor_columns[1])
            axes[1, 0].set_title('Steps vs No Steps')
            axes[1, 0].legend()
        
        # 4. Class imbalance visualization
        label_counts = df["Label"].value_counts()
        axes[1, 1].bar(range(len(label_counts)), label_counts.values)
        axes[1, 1].set_xticks(range(len(label_counts)))
        axes[1, 1].set_xticklabels(label_counts.index, rotation=45)
        axes[1, 1].set_title('Class Imbalance (Linear Scale)')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = "dataset_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
        print("   This is optional - analysis can continue without plots")


def main():
    """Main analysis function."""
    print("🔍 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 60)
    print("This analysis will help identify why the model predicts 99%+ 'No Step'")
    print("=" * 60)
    
    try:
        # Load the dataset
        print("📂 Loading dataset...")
        data_dir = Path("data/raw")
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            print("Please ensure the data/raw directory exists with your CSV files")
            return
        
        df = load_step_data(str(data_dir))
        print(f"✅ Dataset loaded successfully: {len(df):,} samples")
        
        # Run all analyses
        numeric_cols = analyze_dataset_structure(df)
        label_counts = analyze_label_distribution(df)
        analyze_sensor_data_quality(df, numeric_cols)
        issues_found = check_data_consistency(df, numeric_cols)
        analyze_step_patterns(df)
        
        # Generate recommendations
        recommendations = generate_recommendations(label_counts, issues_found)
        
        # Create visualizations
        create_visualization(df, numeric_cols)
        
        # Summary
        print(f"\n🎯 ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Total issues identified: {len(issues_found)}")
        print(f"Recommendations generated: {len(recommendations)}")
        
        print(f"\n📝 NEXT STEPS:")
        print("1. Run the retrain script with class balancing")
        print("2. Optimize detection thresholds")
        print("3. Test with real-world data")
        print("4. Monitor model performance metrics")
        
        return df, recommendations
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []


if __name__ == "__main__":
    df, recommendations = main()


