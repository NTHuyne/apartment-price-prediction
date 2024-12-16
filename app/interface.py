import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
import os

preprocessor = joblib.load("../model/feature_engineering/preprocessor.pkl")
linear_regression_model = joblib.load("../model/models/LinearRegression.pkl")
svr_model = joblib.load("../model/models/SVR.pkl")
xgboost_model = joblib.load("../model/models/XGBoost.pkl")
with open("../data/duAn.json", "r", encoding="utf-8") as f:
    project_data = json.load(f)

st.title("🏢 Apartment Price Prediction App")
st.markdown("### Predict apartment prices in Hanoi with ease!")

st.sidebar.header("User Input Parameters")

location_data = {
    "Quận Ba Đình": {
        "", "Phường Cống Vị", "Phường Điện Biên", "Phường Đội Cấn", "Phường Giảng Võ", "Phường Kim Mã", "Phường Liễu Giai", "Phường Ngọc Hà", "Phường Ngọc Khánh", "Phường Nguyễn Trung Trực", "Phường Phúc Xá", "Phường Quán Thánh", "Phường Thành Công", "Phường Trúc Bạch", "Phường Vĩnh Phúc"
    },
    "Quận Hoàn Kiếm": {
        "", "Phường Chương Dương", "Phường Cửa Đông", "Phường Cửa Nam", "Phường Đồng Xuân", "Phường Hàng Bạc", "Phường Hàng Bài", "Phường Hàng Bồ", "Phường Hàng Bông", "Phường Hàng Buồm", "Phường Hàng Đào", "Phường Hàng Gai", "Phường Hàng Mã", "Phường Hàng Trống", "Phường Lý Thái Tổ", "Phường Phan Chu Trinh", "Phường Phúc Tân", "Phường Trần Hưng Đạo", "Phường Tràng Tiền"
    },
    "Quận Tây Hồ": {
        "", "Phường Bưởi", "Phường Nhật Tân", "Phường Phú Thượng", "Phường Quảng An", "Phường Thụy Khuê", "Phường Tứ Liên", "Phường Xuân La", "Phường Yên Phụ"
    },
    "Quận Cầu Giấy": {
        "", "Phường Dịch Vọng", "Phường Dịch Vọng Hậu", "Phường Mai Dịch", "Phường Nghĩa Đô", "Phường Nghĩa Tân", "Phường Quan Hoa", "Phường Trung Hòa", "Phường Yên Hòa"
    },
    "Quận Đống Đa": {
        "", "Phường Cát Linh", "Phường Hàng Bột", "Phường Khâm Thiên", "Phường Khương Thượng", "Phường Kim Liên", "Phường Láng Hạ", "Phường Láng Thượng", "Phường Nam Đồng", "Phường Ngã Tư Sở", "Phường Ô Chợ Dừa", "Phường Phương Liên", "Phường Phương Mai", "Phường Quang Trung", "Phường Quốc Tử Giám", "Phường Thịnh Quang", "Phường Thổ Quan", "Phường Trung Liệt", "Phường Trung Phụng", "Phường Trung Tự", "Phường Văn Chương", "Phường Văn Miếu"
    },
    "Quận Hai Bà Trưng": {
        "", "Phường Bách Khoa", "Phường Bạch Đằng", "Phường Bạch Mai", "Phường Cầu Dền", "Phường Đống Mác", "Phường Đồng Nhân", "Phường Đồng Tâm", "Phường Lê Đại Hành", "Phường Minh Khai", "Phường Nguyễn Du", "Phường Phạm Đình Hổ", "Phường Phố Huế", "Phường Quỳnh Lôi", "Phường Quỳnh Mai", "Phường Thanh Lương", "Phường Thanh Nhàn", "Phường Trương Định", "Phường Vĩnh Tuy"    
    },
    "Quận Hoàng Mai": {
        "", "Phường Đại Kim", "Phường Định Công", "Phường Giáp Bát", "Phường Hoàng Liệt", "Phường Hoàng Văn Thụ", "Phường Lĩnh Nam", "Phường Mai Động", "Phường Tân Mai", "Phường Thanh Trì", "Phường Thịnh Liệt", "Phường Trần Phú", "Phường Tương Mai", "Phường Vĩnh Hưng", "Phường Yên Sở"    
    },
    "Quận Thanh Xuân": {
        "", "Phường Hạ Đình", "Phường Khương Đình", "Phường Khương Mai", "Phường Khương Trung", "Phường Kim Giang", "Phường Nhân Chính", "Phường Phương Liệt", "Phường Thanh Xuân Bắc", "Phường Thanh Xuân Nam", "Phường Thanh Xuân Trung", "Phường Thượng Đình"    
    },
    "Quận Long Biên": {
        "", "Phường Bồ Đề", "Phường Cự Khối", "Phường Đức Giang", "Phường Gia Thụy", "Phường Giang Biên", "Phường Long Biên", "Phường Ngọc Lâm", "Phường Ngọc Thụy", "Phường Phúc Đồng", "Phường Phúc Lợi", "Phường Sài Đồng", "Phường Thạch Bàn", "Phường Thượng Thanh", "Phường Việt Hưng"    
    },
    "Quận Bắc Từ Liêm": {
        "", "Phường Cổ Nhuế 1", "Phường Cổ Nhuế 2", "Phường Đông Ngạc", "Phường Đức Thắng", "Phường Liên Mạc", "Phường Minh Khai", "Phường Phú Diễn", "Phường Phúc Diễn", "Phường Tây Tựu", "Phường Thượng Cát", "Phường Thụy Phương", "Phường Xuân Đỉnh", "Phường Xuân Tảo"    
    },
    "Quận Nam Từ Liêm": {
        "", "Phường Cầu Diễn", "Phường Đại Mỗ", "Phường Mễ Trì", "Phường Mỹ Đình 1", "Phường Mỹ Đình 2", "Phường Phú Đô", "Phường Phương Canh", "Phường Tây Mỗ", "Phường Trung Văn", "Phường Xuân Phương"
    },
    "Quận Hà Đông": {
        "", "Phường Biên Giang", "Phường Đồng Mai", "Phường Dương Nội", "Phường Hà Cầu", "Phường Kiến Hưng", "Phường La Khê", "Phường Mộ Lao", "Phường Nguyễn Trãi", "Phường Phú La", "Phường Phú Lãm", "Phường Phú Lương", "Phường Phúc La", "Phường Quang Trung", "Phường Vạn Phúc", "Phường Văn Quán", "Phường Yên Nghĩa", "Phường Yết Kiêu"    
    },
    "Thị xã Sơn Tây": {
        "", "Phường Lê Lợi", "Phường Ngô Quyền", "Phường Phú Thịnh", "Phường Quang Trung", "Phường Sơn Lộc", "Phường Trung Hưng", "Phường Trung Sơn Trầm", "Phường Viên Sơn", "Phường Xuân Khanh", "Xã Cổ Đông", "Xã Đường Lâm", "Xã Kim Sơn", "Xã Sơn Đông", "Xã Thanh Mỹ", "Xã Xuân Sơn"    
    },
    "Huyện Sóc Sơn": {
        "", "Thị Trấn Sóc Sơn", "Xã Bắc Phú", "Xã Bắc Sơn", "Xã Đông Xuân", "Xã Đức Hòa", "Xã Hiền Ninh", "Xã Hồng Kỳ", "Xã Kim Lũ", "Xã Mai Đình", "Xã Minh Phú", "Xã Minh Trí", "Xã Nam Sơn", "Xã Phú Cường", "Xã Phù Linh", "Xã Phù Lỗ", "Xã Phú Minh", "Xã Quang Tiến", "Xã Tân Dân", "Xã Tân Hưng", "Xã Tân Minh", "Xã Thạnh Xuân", "Xã Tiên Dược", "Xã Trung Giã", "Xã Việt Long", "Xã Xuân Giang", "Xã Xuân Thu"    
    },
    "Huyện Đông Anh": {
        "", "Thị Trấn Đông Anh", "Xã Bắc Hồng", "Xã Cổ Loa", "Xã Đại Mạch", "Xã Đông Hội", "Xã Dục Tú", "Xã Hải Bối", "Xã Kim Chung", "Xã Kim Nỗ", "Xã Liên Hà", "Xã Mai Lâm", "Xã Nam Hồng", "Xã Nguyên Khê", "Xã Tàm Xá", "Xã Thụy Lâm", "Xã Tiên Dương", "Xã Uy Nỗ", "Xã Vân Hà", "Xã Vân Nội", "Xã Việt Hùng", "Xã Vĩnh Ngọc", "Xã Võng La", "Xã Xuân Canh", "Xã Xuân Nộn"    
    },
    "Huyện Gia Lâm": {
        "", "Thị Trấn Trâu Quỳ", "Thị Trấn Yên Viên", "Xã Bát Tràng", "Xã Cổ Bi", "Xã Đa Tốn", "Xã Đặng Xá", "Xã Phú Thị", "Xã Đông Dư", "Xã Dương Hà", "Xã Dương Quang", "Xã Dương Xá", "Xã Kiêu Kỵ", "Xã Kim Lan", "Xã Văn Đức", "Xã Lệ Chi", "Xã Ninh Hiệp", "Xã Đình Xuyên", "Xã Phù Đổng", "Xã Trung Mầu", "Xã Yên Thường", "Xã Yên Viên", "Xã Kim Sơn"    
    },
    "Huyện Mê Linh": {
        "", "Thị Trấn Chi Đông", "Thị Trấn Quang Minh", "Xã Chu Phan", "Xã Đại Thịnh", "Xã Mê Linh", "Xã Hoàng Kim", "Xã Kim Hoa", "Xã Liên Mạc", "Xã Tam Đồng", "Xã Thạch Đà", "Xã Thanh Lâm", "Xã Tiền Phong", "Xã Tiến Thắng", "Xã Tiến Thịnh", "Xã Tráng Việt", "Xã Tự Lập", "Xã Văn Khê", "Xã Vạn Yên"    
    },
    "Huyện Thanh Trì": {
        "", "Thị Trấn Văn Điển", "Xã Đại Áng", "Xã Đông Mỹ", "Xã Duyên Hà", "Xã Hữu Hòa", "Xã Liên Ninh", "Xã Ngọc Hồi", "Xã Ngũ Hiệp", "Xã Tả Thanh Oai", "Xã Tam Hiệp", "Xã Tân Triều", "Xã Thanh Liệt", "Xã Tứ Hiệp", "Xã Vạn Phúc", "Xã Vĩnh Quỳnh", "Xã Yên Mỹ"    
    },
    "Huyện Phúc Thọ": {
        "", "Thị Trấn Phúc Thọ", "Xã Hát Môn", "Xã Hiệp Thuận", "Xã Liên Hiệp", "Xã Long Xuyên", "Xã Ngọc Tảo", "Xã Phúc Hòa", "Xã Phụng Thượng", "Xã Sen Phương", "Xã Tam Hiệp", "Xã Tam Thuấn", "Xã Thanh Đa", "Xã Thọ Lộc", "Xã Thượng Cốc", "Xã Tích Giang", "Xã Trạch Mỹ Lộc", "Xã Vân Hà", "Xã Vân Nam", "Xã Vân Phúc", "Xã Võng Xuyên", "Xã Xuân Đình"    
    },
    "Huyện Ba Vì": {
        "", "Thị Trấn Tây Đằng", "Xã Ba Trại", "Xã Ba Vì", "Xã Cẩm Lĩnh", "Xã Cam Thượng", "Xã Châu Sơn", "Xã Chu Minh", "Xã Cổ Đô", "Xã Đông Quang", "Xã Đồng Thái", "Xã Khánh Thượng", "Xã Minh Châu", "Xã Minh Quang", "Xã Phong Vân", "Xã Phú Châu", "Xã Phú Cường", "Xã Phú Đông", "Xã Phú Phương", "Xã Phú Sơn", "Xã Sơn Đà", "Xã Tản Hồng", "Xã Tản Lĩnh", "Xã Thái Hòa", "Xã Thuần Mỹ", "Xã Thụy An", "Xã Tiên Phong", "Xã Tòng Bạt", "Xã Vân Hòa", "Xã Vạn Thắng", "Xã Vật Lại", "Xã Yên Bài"    
    },
    "Huyện Đan Phượng": {
        "", "Thị Trấn Phùng", "Xã Đan Phượng", "Xã Đồng Tháp", "Xã Hạ Mỗ", "Xã Hồng Hà", "Xã Liên Hà", "Xã Liên Hồng", "Xã Liên Trung", "Xã Phương Đình", "Xã Song Phượng", "Xã Tân Hội", "Xã Tân Lập", "Xã Thọ An", "Xã Thọ Xuân", "Xã Thượng Mỗ", "Xã Trung Châu"    
    },
    "Huyện Quốc Oai": {
        "", "Thị Trấn Quốc Oai", "Xã Cấn Hữu", "Xã Cộng Hòa", "Xã Đại Thành", "Xã Đồng Quang", "Xã Đông Yên", "Xã Hòa Thạch", "Xã Liệp Tuyết", "Xã Nghĩa Hương", "Xã Ngọc Liệp", "Xã Ngọc Mỹ", "Xã Phú Cát", "Xã Phú Mãn", "Xã Phượng Cách", "Xã Sài Sơn", "Xã Tân Hòa", "Xã Tân Phú", "Xã Thạch Thán", "Xã Tuyết Nghĩa", "Xã Yên Sơn", "Xã Đông Xuân"
    },
    "Huyện Hoài Đức": {
        "", "Thị Trấn Trạm Trôi", "Xã An Khánh", "Xã An Thượng", "Xã Cát Quế", "Xã Đắc Sở", "Xã Di Trạch", "Xã Đông La", "Xã Đức Giang", "Xã Đức Thượng", "Xã Dương Liễu", "Xã Kim Chung", "Xã La Phù", "Xã Lại Yên", "Xã Minh Khai", "Xã Sơn Đồng", "Xã Song Phương", "Xã Tiền Yên", "Xã Vân Canh", "Xã Vân Côn", "Xã Yên Sở"    
    },
    "Huyện Thạch Thất": {
        "", "Thị Trấn Liên Quan", "Xã Bình Phú", "Xã Bình Yên", "Xã Cẩm Yên", "Xã Cần Kiệm", "Xã Canh Nậu", "Xã Chàng Sơn", "Xã Đại Đồng", "Xã Dị Nậu", "Xã Đồng Trúc", "Xã Hạ Bằng", "Xã Hương Ngải", "Xã Hữu Bằng", "Xã Kim Quan", "Xã Lại Thượng", "Xã Phú Kim", "Xã Phùng Xá", "Xã Tân Xã", "Xã Thạch Hòa", "Xã Thạch Xá", "Xã Tiến Xuân", "Xã Yên Bình", "Xã Yên Trung"
    },
    "Huyện Thanh Oai": {
        "", "Thị Trấn Kim Bài", "Xã Bích Hòa", "Xã Bình Minh", "Xã Cao Dương", "Xã Cao Viên", "Xã Cự Khê", "Xã Dân Hòa", "Xã Đỗ Động", "Xã Hồng Dương", "Xã Kim An", "Xã Kim Thư", "Xã Liên Châu", "Xã Mỹ Hưng", "Xã Phương Trung", "Xã Tam Hưng", "Xã Tân Ước", "Xã Thanh Cao", "Xã Thanh Mai", "Xã Thanh Thùy", "Xã Thanh Văn", "Xã Xuân Dương"    
    },
    "Huyện Chương Mỹ": {
        "", "Thị Trấn Chúc Sơn", "Thị Trấn Xuân Mai", "Xã Đại Yên", "Xã Đông Phương Yên", "Xã Đông Sơn", "Xã Đồng Lạc", "Xã Đồng Phú", "Xã Hòa Chính", "Xã Hoàng Diệu", "Xã Hoàng Văn Thụ", "Xã Hồng Phong", "Xã Hợp Đồng", "Xã Hữu Văn", "Xã Lam Điền", "Xã Mỹ Lương", "Xã Nam Phương Tiến", "Xã Ngọc Hòa", "Xã Phú Nam An", "Xã Phú Nghĩa", "Xã Phụng Châu", "Xã Quảng Bị", "Xã Tân Tiến", "Xã Tiên Phương", "Xã Tốt Động", "Xã Thanh Bình", "Xã Thủy Xuân Tiên", "Xã Thụy Hương", "Xã Thượng Vực", "Xã Trần Phú", "Xã Trung Hòa", "Xã Trường Yên", "Xã Văn Võ"    
    },
    "Huyện Thường Tín": {
        "", "Thị Trấn Thường Tín", "Xã Chương Dương", "Xã Dũng Tiến", "Xã Duyên Thái", "Xã Hà Hồi", "Xã Hiền Giang", "Xã Hòa Bình", "Xã Khánh Hà", "Xã Hồng Vân", "Xã Lê Lợi", "Xã Liên Phương", "Xã Minh Cường", "Xã Nghiêm Xuyên", "Xã Nguyễn Trãi", "Xã Nhị Khê", "Xã Ninh Sở", "Xã Quất Động", "Xã Tân Minh", "Xã Thắng Lợi", "Xã Thống Nhất", "Xã Thư Phú", "Xã Tiền Phong", "Xã Tô Hiệu", "Xã Tự Nhiên", "Xã Vạn Điểm", "Xã Văn Bình", "Xã Văn Phú", "Xã Văn Tự", "Xã Vân Tảo"
    },
    "Huyện Mỹ Đức": {
        "", "Thị Trấn Đại Nghĩa", "Xã An Mỹ", "Xã An Phú", "Xã An Tiến", "Xã Bột Xuyên", "Xã Đại Hưng", "Xã Đốc Tín", "Xã Đồng Tâm", "Xã Hồng Sơn", "Xã Hợp Thanh", "Xã Hợp Tiến", "Xã Hùng Tiến", "Xã Hương Sơn", "Xã Lê Thanh", "Xã Mỹ Thành", "Xã Phù Lưu Tế", "Xã Phúc Lâm", "Xã Phùng Xá", "Xã Thượng Lâm", "Xã Tuy Lai", "Xã Vạn Kim", "Xã Xuy Xá"    
    },
    "Huyện Phú Xuyên": {
        "", "Thị Trấn Phú Xuyên", "Thị Trấn Phú Minh", "Xã Bạch Hạ", "Xã Châu Can", "Xã Chuyên Mỹ", "Xã Đại Thắng", "Xã Đại Xuyên", "Xã Hoàng Long", "Xã Hồng Minh", "Xã Hồng Thái", "Xã Khai Thái", "Xã Minh Tân", "Xã Nam Phong", "Xã Nam Tiến", "Xã Nam Triều", "Xã Phú Túc", "Xã Phú Yên", "Xã Phúc Tiến", "Xã Phượng Dực", "Xã Quang Lãng", "Xã Quang Trung", "Xã Sơn Hà", "Xã Tân Dân", "Xã Tri Thủy", "Xã Tri Trung", "Xã Văn Hoàng", "Xã Vân Từ"    
    },
    "Huyện Ứng Hòa": {
        "", "Thị Trấn Vân Đình", "Xã Cao Thành", "Xã Đại Cường", "Xã Đại Hùng", "Xã Đội Bình", "Xã Đông Lỗ", "Xã Đồng Tiến", "Xã Đồng Tân", "Xã Hoa Sơn", "Xã Hòa Lâm", "Xã Hòa Nam", "Xã Hòa Phú", "Xã Hòa Xá", "Xã Hồng Quang", "Xã Kim Đường", "Xã Liên Bạt", "Xã Lưu Hoàng", "Xã Minh Đức", "Xã Phù Lưu", "Xã Phương Tú", "Xã Quảng Phú Cầu", "Xã Sơn Công", "Xã Tảo Dương Văn", "Xã Trầm Lộng", "Xã Trung Tú", "Xã Trường Thịnh", "Xã Vạn Thái", "Xã Viên An", "Xã Viên Nội"
    }
}
district = st.sidebar.selectbox("Quận/Huyện", list(location_data.keys()))
sub_district = st.sidebar.selectbox("Phường/Xã", sorted(location_data[district]))

if sub_district == '':
    projects = []
    for precinct in project_data.get(district, {}).values():
        projects.extend(precinct)
else:
    projects = project_data.get(district, {}).get(sub_district, [])
projects = sorted(set(projects))
du_an = st.sidebar.selectbox("Dự án", projects)
acreage_value = st.sidebar.number_input("Diện tích (m2)", min_value=0.0, value=50.0)
huong = st.sidebar.selectbox("Hướng", ["", "Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"])
phap_ly = st.sidebar.selectbox("Pháp lý", ["", "Giấy chứng nhận quyền sở hữu đất", "Giấy tờ hợp lệ"])
no_bed = st.sidebar.number_input("Số phòng ngủ", min_value=-1, value=-1)
no_bathroom = st.sidebar.number_input("Số phòng tắm", min_value=-1, value=-1)
so_lau = st.sidebar.number_input("Số lầu", min_value=-1, value=-1)

location_str = f"{district} - {sub_district}"

if st.sidebar.button("🏡 Predict Price"):
    lstm_model = load_model("../model/models/lstm_apartment_model.h5", compile=False)
    if du_an == '':
        du_an = np.nan
    if huong == '':
        huong = np.nan
    if phap_ly == '':
        phap_ly = np.nan
    if sub_district == '':
        sub_district = np.nan
    data = {
        'duAn': [du_an],
        'huong': [huong],
        'phapLy': [phap_ly],
        'noBed': [no_bed],
        'soLau': [so_lau],
        'Precinct': [sub_district],
        'District': [district],
        'acreage_value': [acreage_value],
        'noBathroom': [no_bathroom],
    }
    print(data)
    df = pd.DataFrame(data)
    print(df)
    input_params= preprocessor.transform(df)    
    predicted_price_1 = linear_regression_model.predict(input_params)[0]
    print("Done")
    predicted_price_2 = svr_model.predict(input_params)[0]
    print("Done")
    predicted_price_3 = xgboost_model.predict(input_params)[0]
    print("Done")
    timesteps = 1  # Define timesteps
    input_params_new = input_params.toarray()
    n_samples, n_features = input_params_new.shape
    n_sequences = n_samples // timesteps  # Ensure divisible by timesteps

    # Reshape to (samples, timesteps, features)
    X_lstm = input_params_new[:n_sequences * timesteps, :].reshape(n_sequences, timesteps, n_features)
    predicted_price_4 = lstm_model.predict(X_lstm)[0][0]
    
    st.success(f"💰 Predicted Prices for {location_str}:")
    st.write(f"Linear Regression Model: {predicted_price_1:,.2f} billion VND")
    st.write(f"SVR Model: {predicted_price_2:,.2f} billion VND")
    st.write(f"XGBoost Model: {predicted_price_3:,.2f} billion VND")
    st.write(f"LSTM Model: {predicted_price_4:,.2f} billion VND")