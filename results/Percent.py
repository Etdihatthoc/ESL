import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/results/test_predictions.csv")  # Replace with your actual file path

# Round predict score to nearest 0.5
df["Predict Score Rounded"] = np.round(df["Predict Score"] * 2) / 2

# Calculate absolute error
df["Absolute Error"] = np.abs(df["GroundTruth"] - df["Predict Score Rounded"])

# Calculate total samples
total_samples = len(df)

# Calculate percentages
exact_match = len(df[df["Absolute Error"] == 0])
within_0_5 = len(df[df["Absolute Error"] <= 0.5])
within_1_0 = len(df[df["Absolute Error"] <= 1.0])
greater_1_0 = len(df[df["Absolute Error"] > 1.0])

# Convert to percentages
exact_match_pct = (exact_match / total_samples) * 100
within_0_5_pct = (within_0_5 / total_samples) * 100
within_1_0_pct = (within_1_0 / total_samples) * 100
greater_1_0_pct = (greater_1_0 / total_samples) * 100

# Print results
print("="*50)
print("PREDICTION ACCURACY ANALYSIS")
print("="*50)
print(f"Total samples: {total_samples}")
print(f"Exact match (error = 0.0): {exact_match} ({exact_match_pct:.2f}%)")
print(f"Within 0.5 (error ≤ 0.5): {within_0_5} ({within_0_5_pct:.2f}%)")
print(f"Within 1.0 (error ≤ 1.0): {within_1_0} ({within_1_0_pct:.2f}%)")
print(f"Greater than 1.0 (error > 1.0): {greater_1_0} ({greater_1_0_pct:.2f}%)")
print("="*50)

# Additional breakdown
print("\nDETAILED BREAKDOWN:")
print(f"Exact match: {exact_match_pct:.2f}%")
print(f"Within 0.5 but not exact: {(within_0_5 - exact_match) / total_samples * 100:.2f}%")
print(f"Within 1.0 but not 0.5: {(within_1_0 - within_0_5) / total_samples * 100:.2f}%")
print(f"Error > 1.0: {greater_1_0_pct:.2f}%")

# Verify percentages sum to 100%
print(f"\nVerification: {within_1_0_pct + greater_1_0_pct:.2f}% (should be 100%)")