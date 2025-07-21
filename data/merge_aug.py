import pandas as pd

# 1. Đọc dữ liệu
df1 = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/data/Full/train_pro.csv")
df2 = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/data/high_train_dataset_augment.csv")
output_path = "merged.csv"

# 2. Xác định tập cột chung: 
#    - Trước hết lấy set các cột của file2 (ngoại trừ text, augmented_text)
cols2 = set(df2.columns)

# 3. Trên df1: xóa các cột không có trong file2
drop_cols = [c for c in df1.columns if c not in cols2]
df1 = df1.drop(columns=drop_cols)

# 4. Trên df2:
#    - Xóa cột 'text'
#    - Đổi tên 'augmented_text' thành 'text'
df2 = df2.drop(columns=["text"], errors="ignore")
df2 = df2.rename(columns={"augmented_text": "text"})

# 5. Giờ hai DataFrame có cùng columns (cột của df1 sau bước 3)
#    Nếu df2 thiếu cột nào (ví dụ final), pandas sẽ tự điền NaN
#    Sắp xếp lại thứ tự cột theo df1
common_cols = df1.columns.tolist()
df2 = df2.reindex(columns=common_cols)

# 6. Nối hai DataFrame lại
df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)

# 7. Xuất ra file mới
df_merged.to_csv(output_path, index=False)

print(f"Đã merged và lưu thành: {output_path}")
