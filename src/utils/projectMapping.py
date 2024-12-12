import pandas as pd
import json

# Load dữ liệu training
data = pd.read_csv("C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\data\\normalized\\clean_dataset4.csv")

# Xử lý giá trị NaN trong cột Precinct và Street
data['Precinct'] = data['Precinct'].fillna('')
data['duAn'] = data['duAn'].fillna('')

# Tạo mapping
mapping = {}
for _, row in data.iterrows():
    district = row['District']
    precinct = row['Precinct']
    project = row['duAn']  # Giả sử cột 'Street' là tên dự án, thay đổi nếu cần

    # Khởi tạo quận/huyện nếu chưa tồn tại
    if district not in mapping:
        mapping[district] = {}

    # Khởi tạo phường/xã nếu chưa tồn tại
    if precinct not in mapping[district]:
        mapping[district][precinct] = []

    # Thêm dự án vào danh sách (nếu chưa tồn tại)
    if project not in mapping[district][precinct]:
        mapping[district][precinct].append(project)

# Sắp xếp dữ liệu trong mapping
sorted_mapping = {
    district: {
        precinct: sorted(projects)  # Sắp xếp danh sách dự án
        for precinct, projects in sorted(precincts.items())  # Sắp xếp theo phường/xã
    }
    for district, precincts in sorted(mapping.items())  # Sắp xếp theo quận/huyện
}

# Lưu mapping ra file JSON
with open("C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\data\\duAn.json", "w", encoding="utf-8") as f:
    json.dump(sorted_mapping, f, ensure_ascii=False, indent=4)

print("Mapping đã được tạo và lưu thành công.")

