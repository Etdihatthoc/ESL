# import pandas as pd

# # 1. Đọc file train_pro_WhisperSmall.csv
# df_small = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/data/Full/train_pro_WhisperSmall.csv")

# # 2. Xóa cột 'text' cũ nếu tồn tại
# if 'text' in df_small.columns:
#     df_small = df_small.drop(columns=['text'])

# # 3. Đổi tên 'text_small' thành 'text'
# #    Nếu tên cột của bạn khác (ví dụ 'transcript'), hãy thay 'text_small' tương ứng.
# df_small = df_small.rename(columns={'text_small': 'text'})

# # 4. Đọc file gốc train_pro.csv
# df_base = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/data/Full/merged.csv")

# # 5. Nối hai DataFrame (không loại bỏ dòng trùng lặp)
# df_full = pd.concat([df_base, df_small], ignore_index=True)

# # 6. Lưu ra file mới
# output_path = "/media/gpus/Data/AES/ESL-Grading/data/Full/Full_train.csv"
# df_full.to_csv(output_path, index=False)

# print(f"Đã sinh file: {output_path} với {len(df_full)} dòng tổng cộng.")


import pandas as pd

# 1. Đọc file train_pro_WhisperSmall.csv
df_small = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/data/Full/generated_results.csv")


df_small = df_small.rename(columns={'part': 'question_type'})
df_small = df_small.rename(columns={'answer': 'text'})

# 4. Đọc file gốc train_pro.csv
df_base = pd.read_csv("/media/gpus/Data/AES/ESL-Grading/data/Full/Full_train.csv")

# 5. Nối hai DataFrame (không loại bỏ dòng trùng lặp)
df_full = pd.concat([df_base, df_small], ignore_index=True)

# 6. Lưu ra file mới
output_path = "/media/gpus/Data/AES/ESL-Grading/data/Full/Full_train_aug.csv"
df_full.to_csv(output_path, index=False)

print(f"Đã sinh file: {output_path} với {len(df_full)} dòng tổng cộng.")
