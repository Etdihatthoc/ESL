import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. Đọc CSV
df = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/results/test_predictions.csv")

# 2. Làm tròn cột Predict Score để có điểm rời rạc (0.5 increments)
df["Predict Score"] = np.round(df["Predict Score"] * 2) / 2
df["Predict Score Rounded"] = np.round(df["Predict Score"] * 2) / 2
df["Absolute Error"] = np.abs(df["GroundTruth"] - df["Predict Score Rounded"])
# 3. Tính MSE và MAE tổng thể
mse_overall = mean_squared_error(df["GroundTruth"], df["Predict Score Rounded"])
mae_overall = mean_absolute_error(df["GroundTruth"], df["Predict Score Rounded"])

print("="*50)
print("METRICS EVALUATION REPORT")
print("="*50)
print(f"📊 OVERALL METRICS:")
print(f"MSE: {mse_overall:.4f}")
print(f"MAE: {mae_overall:.4f}")
print(f"Total samples: {len(df)}")

# 4. Tính metrics theo range điểm
print(f"\n📊 METRICS BY SCORE RANGES:")

# Range [3-7.5]
range_3_75 = df[(df["GroundTruth"] >= 3) & (df["GroundTruth"] <= 7.5)]
if len(range_3_75) > 0:
    mse_3_75 = mean_squared_error(range_3_75["GroundTruth"], range_3_75["Predict Score Rounded"])
    mae_3_75 = mean_absolute_error(range_3_75["GroundTruth"], range_3_75["Predict Score Rounded"])
    print(f"Range [3.0-7.5]: MSE={mse_3_75:.4f}, MAE={mae_3_75:.4f}, Count={len(range_3_75)}")

# Range [8-10]
range_8_10 = df[(df["GroundTruth"] >= 8) & (df["GroundTruth"] <= 10)]
if len(range_8_10) > 0:
    mse_8_10 = mean_squared_error(range_8_10["GroundTruth"], range_8_10["Predict Score Rounded"])
    mae_8_10 = mean_absolute_error(range_8_10["GroundTruth"], range_8_10["Predict Score Rounded"])
    print(f"Range [8.0-10.0]: MSE={mse_8_10:.4f}, MAE={mae_8_10:.4f}, Count={len(range_8_10)}")

# 5. Tính metrics theo từng điểm cụ thể từ 3 đến 10
print(f"\n📊 METRICS BY INDIVIDUAL SCORES:")
all_scores = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
for score in all_scores:
    score_data = df[df["GroundTruth"] == score]
    if len(score_data) > 0:
        mse_score = mean_squared_error(score_data["GroundTruth"], score_data["Predict Score Rounded"])
        mae_score = mean_absolute_error(score_data["GroundTruth"], score_data["Predict Score Rounded"])
        print(f"Score {score}: MSE={mse_score:.4f}, MAE={mae_score:.4f}, Count={len(score_data)}")

# 6. Top 10 dự đoán lệch xa nhất
print(f"\n📊 TOP 10 WORST PREDICTIONS:")
worst_predictions = df.nlargest(10, 'Absolute Error')[['GroundTruth', 'Predict Score', 'Absolute Error']]
print(worst_predictions.round(3).to_string(index=False))

# 7. Vẽ và lưu Confusion Matrix
plt.figure(figsize=(10, 8))
# Tạo confusion matrix cho range 3-10
score_range = np.arange(3, 10.5, 0.5)
confusion_data = np.zeros((len(score_range), len(score_range)))

for i, gt_score in enumerate(score_range):
    for j, pred_score in enumerate(score_range):
        count = len(df[(df["GroundTruth"] == gt_score) & (df["Predict Score Rounded"] == pred_score)])
        confusion_data[i, j] = count

sns.heatmap(confusion_data, annot=True, fmt='g', cmap='Blues', 
            xticklabels=[f'{x:.1f}' for x in score_range], 
            yticklabels=[f'{x:.1f}' for x in score_range])
plt.title('Confusion Matrix (Score Range 3-10)', fontweight='bold', fontsize=14)
plt.xlabel('Predicted Score', fontsize=12)
plt.ylabel('Ground Truth', fontsize=12)
plt.tight_layout()
plt.savefig('/media/gpus/Data/AES/ESL-Grading/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Vẽ biểu đồ chênh lệch theo từng mức điểm
plt.figure(figsize=(12, 6))

# Tính error cho từng điểm
score_points = []
positive_errors = []  # Dự đoán > thật
negative_errors = []  # Dự đoán < thật

for score in all_scores:
    score_data = df[df["GroundTruth"] == score]
    if len(score_data) > 0:
        errors = score_data["Predict Score Rounded"] - score_data["GroundTruth"]
        pos_error = errors[errors > 0].mean() if len(errors[errors > 0]) > 0 else 0
        neg_error = errors[errors < 0].mean() if len(errors[errors < 0]) > 0 else 0
        
        score_points.append(score)
        positive_errors.append(pos_error)
        negative_errors.append(neg_error)

# Vẽ đường thẳng cho các điểm 3-10
plt.plot(score_points, [0] * len(score_points), 'k-', linewidth=2, label='Perfect Prediction')

# Vẽ cột thể hiện chênh lệch
width = 0.1
for i, score in enumerate(score_points):
    if positive_errors[i] > 0:
        plt.bar(score, positive_errors[i], width, color='red', alpha=0.7, label='Overestimation' if i == 0 else "")
    if negative_errors[i] < 0:
        plt.bar(score, negative_errors[i], width, color='blue', alpha=0.7, label='Underestimation' if i == 0 else "")

plt.xlabel('Ground Truth Score', fontsize=12)
plt.ylabel('Mean Prediction Error', fontsize=12)
plt.title('Prediction Error by Score Level', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(score_points)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Thêm text annotation cho các giá trị
for i, score in enumerate(score_points):
    if positive_errors[i] > 0:
        plt.text(score, positive_errors[i] + 0.02, f'{positive_errors[i]:.3f}', 
                ha='center', va='bottom', fontsize=9)
    if negative_errors[i] < 0:
        plt.text(score, negative_errors[i] - 0.02, f'{negative_errors[i]:.3f}', 
                ha='center', va='top', fontsize=9)

plt.tight_layout()
plt.savefig('/media/gpus/Data/AES/ESL-Grading/results/error_by_score.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Saved plots to:")
print(f"- /media/gpus/Data/AES/ESL-Grading/results/confusion_matrix.png")
print(f"- /media/gpus/Data/AES/ESL-Grading/results/error_by_score.png")

print("\n" + "="*50)
print("EVALUATION COMPLETED")
print("="*50)