import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Giả lập dữ liệu giao dịch (Transaction Data)
# Phần lớn là giao dịch bình thường (số tiền nhỏ/trung bình)
rng = np.random.RandomState(42)
X_normal = 0.3 * rng.randn(100, 2)
X_train = np.r_[X_normal + 2, X_normal - 2]

# Tạo ra vài giao dịch "bất thường" (số tiền cực lớn hoặc lạ)
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# Gộp dữ liệu lại
X = np.r_[X_train, X_outliers]
df = pd.DataFrame(X, columns=['Amount_Normalized', 'Frequency_Normalized'])

# 2. Huấn luyện mô hình Isolation Forest
# contamination=0.1 nghĩa là ta ước tính có khoảng 10% gian lận
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(df)

# 3. Dự đoán (1: Bình thường, -1: Gian lận)
df['Status'] = clf.predict(df[['Amount_Normalized', 'Frequency_Normalized']])
df['Risk_Label'] = df['Status'].apply(lambda x: 'Normal' if x == 1 else 'FRAUD DETECTED')

# 4. Xuất báo cáo các giao dịch gian lận
frauds = df[df['Status'] == -1]
print(f"=== SECURITY ALERT: FOUND {len(frauds)} SUSPICIOUS TRANSACTIONS ===")
print(frauds.head())
