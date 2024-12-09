import streamlit as st
import joblib
import numpy as np
import pandas as pd

preprocessor = joblib.load("model/feature_engineering/preprocessor.pkl")
linear_regression_model = joblib.load("model/models/LinearRegression.pkl")
svr_model = joblib.load("model/models/SVR.pkl")
xgboost_model = joblib.load("model/models/XGBoost.pkl")

st.title("🏢 Apartment Price Prediction App")
st.markdown("### Predict apartment prices in Hanoi with ease!")

st.sidebar.header("User Input Parameters")

location_data = {
    "Quận Ba Đình": {
        "Phường Cống Vị", "Phường Điện Biên", "Phường Đội Cấn", "Phường Giảng Võ", "Phường Kim Mã", "Phường Liễu Giai", "Phường Ngọc Hà", "Phường Ngọc Khánh", "Phường Nguyễn Trung Trực", "Phường Phúc Xá", "Phường Quán Thánh", "Phường Thành Công", "Phường Trúc Bạch", "Phường Vĩnh Phúc"
    },
    "Quận Hoàn Kiếm": {
        "Phường Chương Dương", "Phường Cửa Đông", "Phường Cửa Nam", "Phường Đồng Xuân", "Phường Hàng Bạc", "Phường Hàng Bài", "Phường Hàng Bồ", "Phường Hàng Bông", "Phường Hàng Buồm", "Phường Hàng Đào", "Phường Hàng Gai", "Phường Hàng Mã", "Phường Hàng Trống", "Phường Lý Thái Tổ", "Phường Phan Chu Trinh", "Phường Phúc Tân", "Phường Trần Hưng Đạo", "Phường Tràng Tiền"
    },
    "Quận Tây Hồ": {
        "Phường Bưởi", "Phường Nhật Tân", "Phường Phú Thượng", "Phường Quảng An", "Phường Thụy Khuê", "Phường Tứ Liên", "Phường Xuân La", "Phường Yên Phụ"
    },
    "Quận Cầu Giấy": {
        "Phường Dịch Vọng", "Phường Dịch Vọng Hậu", "Phường Mai Dịch", "Phường Nghĩa Đô", "Phường Nghĩa Tân", "Phường Quan Hoa", "Phường Trung Hòa", "Phường Yên Hòa"
    },
    "Quận Đống Đa": {
        "Phường Cát Linh", "Phường Hàng Bột", "Phường Khâm Thiên", "Phường Khương Thượng", "Phường Kim Liên", "Phường Láng Hạ", "Phường Láng Thượng", "Phường Nam Đồng", "Phường Ngã Tư Sở", "Phường Ô Chợ Dừa", "Phường Phương Liên", "Phường Phương Mai", "Phường Quang Trung", "Phường Quốc Tử Giám", "Phường Thịnh Quang", "Phường Thổ Quan", "Phường Trung Liệt", "Phường Trung Phụng", "Phường Trung Tự", "Phường Văn Chương", "Phường Văn Miếu"
    },
    "Quận Hai Bà Trưng": {
        "Phường Bách Khoa", "Phường Bạch Đằng", "Phường Bạch Mai", "Phường Cầu Dền", "Phường Đống Mác", "Phường Đồng Nhân", "Phường Đồng Tâm", "Phường Lê Đại Hành", "Phường Minh Khai", "Phường Nguyễn Du", "Phường Phạm Đình Hổ", "Phường Phố Huế", "Phường Quỳnh Lôi", "Phường Quỳnh Mai", "Phường Thanh Lương", "Phường Thanh Nhàn", "Phường Trương Định", "Phường Vĩnh Tuy"    
    },
    "Quận Hoàng Mai": {
        "Phường Đại Kim", "Phường Định Công", "Phường Giáp Bát", "Phường Hoàng Liệt", "Phường Hoàng Văn Thụ", "Phường Lĩnh Nam", "Phường Mai Động", "Phường Tân Mai", "Phường Thanh Trì", "Phường Thịnh Liệt", "Phường Trần Phú", "Phường Tương Mai", "Phường Vĩnh Hưng", "Phường Yên Sở"    
    },
    "Quận Thanh Xuân": {
        "Phường Hạ Đình", "Phường Khương Đình", "Phường Khương Mai", "Phường Khương Trung", "Phường Kim Giang", "Phường Nhân Chính", "Phường Phương Liệt", "Phường Thanh Xuân Bắc", "Phường Thanh Xuân Nam", "Phường Thanh Xuân Trung", "Phường Thượng Đình"    
    },
    "Quận Long Biên": {
        "Phường Bồ Đề", "Phường Cự Khối", "Phường Đức Giang", "Phường Gia Thụy", "Phường Giang Biên", "Phường Long Biên", "Phường Ngọc Lâm", "Phường Ngọc Thụy", "Phường Phúc Đồng", "Phường Phúc Lợi", "Phường Sài Đồng", "Phường Thạch Bàn", "Phường Thượng Thanh", "Phường Việt Hưng"    
    },
    "Quận Bắc Từ Liêm": {
        "Phường Cổ Nhuế 1", "Phường Cổ Nhuế 2", "Phường Đông Ngạc", "Phường Đức Thắng", "Phường Liên Mạc", "Phường Minh Khai", "Phường Phú Diễn", "Phường Phúc Diễn", "Phường Tây Tựu", "Phường Thượng Cát", "Phường Thụy Phương", "Phường Xuân Đỉnh", "Phường Xuân Tảo"    
    },
    "Quận Nam Từ Liêm": {
        "Phường Cầu Diễn", "Phường Đại Mỗ", "Phường Mễ Trì", "Phường Mỹ Đình 1", "Phường Mỹ Đình 2", "Phường Phú Đô", "Phường Phương Canh", "Phường Tây Mỗ", "Phường Trung Văn", "Phường Xuân Phương"
    },
    "Quận Hà Đông": {
        "Phường Biên Giang", "Phường Đồng Mai", "Phường Dương Nội", "Phường Hà Cầu", "Phường Kiến Hưng", "Phường La Khê", "Phường Mộ Lao", "Phường Nguyễn Trãi", "Phường Phú La", "Phường Phú Lãm", "Phường Phú Lương", "Phường Phúc La", "Phường Quang Trung", "Phường Vạn Phúc", "Phường Văn Quán", "Phường Yên Nghĩa", "Phường Yết Kiêu"    
    },
    "Thị xã Sơn Tây": {
        "Phường Lê Lợi", "Phường Ngô Quyền", "Phường Phú Thịnh", "Phường Quang Trung", "Phường Sơn Lộc", "Phường Trung Hưng", "Phường Trung Sơn Trầm", "Phường Viên Sơn", "Phường Xuân Khanh", "Xã Cổ Đông", "Xã Đường Lâm", "Xã Kim Sơn", "Xã Sơn Đông", "Xã Thanh Mỹ", "Xã Xuân Sơn"    
    },
    "Huyện Sóc Sơn": {
        "Thị Trấn Sóc Sơn", "Xã Bắc Phú", "Xã Bắc Sơn", "Xã Đông Xuân", "Xã Đức Hòa", "Xã Hiền Ninh", "Xã Hồng Kỳ", "Xã Kim Lũ", "Xã Mai Đình", "Xã Minh Phú", "Xã Minh Trí", "Xã Nam Sơn", "Xã Phú Cường", "Xã Phù Linh", "Xã Phù Lỗ", "Xã Phú Minh", "Xã Quang Tiến", "Xã Tân Dân", "Xã Tân Hưng", "Xã Tân Minh", "Xã Thạnh Xuân", "Xã Tiên Dược", "Xã Trung Giã", "Xã Việt Long", "Xã Xuân Giang", "Xã Xuân Thu"    
    },
    "Huyện Đông Anh": {
        "Thị Trấn Đông Anh", "Xã Bắc Hồng", "Xã Cổ Loa", "Xã Đại Mạch", "Xã Đông Hội", "Xã Dục Tú", "Xã Hải Bối", "Xã Kim Chung", "Xã Kim Nỗ", "Xã Liên Hà", "Xã Mai Lâm", "Xã Nam Hồng", "Xã Nguyên Khê", "Xã Tàm Xá", "Xã Thụy Lâm", "Xã Tiên Dương", "Xã Uy Nỗ", "Xã Vân Hà", "Xã Vân Nội", "Xã Việt Hùng", "Xã Vĩnh Ngọc", "Xã Võng La", "Xã Xuân Canh", "Xã Xuân Nộn"    
    },
    "Huyện Gia Lâm": {
        "Thị Trấn Trâu Quỳ", "Thị Trấn Yên Viên", "Xã Bát Tràng", "Xã Cổ Bi", "Xã Đa Tốn", "Xã Đặng Xá", "Xã Phú Thị", "Xã Đông Dư", "Xã Dương Hà", "Xã Dương Quang", "Xã Dương Xá", "Xã Kiêu Kỵ", "Xã Kim Lan", "Xã Văn Đức", "Xã Lệ Chi", "Xã Ninh Hiệp", "Xã Đình Xuyên", "Xã Phù Đổng", "Xã Trung Mầu", "Xã Yên Thường", "Xã Yên Viên", "Xã Kim Sơn"    
    },
    "Huyện Mê Linh": {
        "Thị Trấn Chi Đông", "Thị Trấn Quang Minh", "Xã Chu Phan", "Xã Đại Thịnh", "Xã Mê Linh", "Xã Hoàng Kim", "Xã Kim Hoa", "Xã Liên Mạc", "Xã Tam Đồng", "Xã Thạch Đà", "Xã Thanh Lâm", "Xã Tiền Phong", "Xã Tiến Thắng", "Xã Tiến Thịnh", "Xã Tráng Việt", "Xã Tự Lập", "Xã Văn Khê", "Xã Vạn Yên"    
    },
    "Huyện Thanh Trì": {
        "Thị Trấn Văn Điển", "Xã Đại Áng", "Xã Đông Mỹ", "Xã Duyên Hà", "Xã Hữu Hòa", "Xã Liên Ninh", "Xã Ngọc Hồi", "Xã Ngũ Hiệp", "Xã Tả Thanh Oai", "Xã Tam Hiệp", "Xã Tân Triều", "Xã Thanh Liệt", "Xã Tứ Hiệp", "Xã Vạn Phúc", "Xã Vĩnh Quỳnh", "Xã Yên Mỹ"    
    },
    "Huyện Phúc Thọ": {
        "Thị Trấn Phúc Thọ", "Xã Hát Môn", "Xã Hiệp Thuận", "Xã Liên Hiệp", "Xã Long Xuyên", "Xã Ngọc Tảo", "Xã Phúc Hòa", "Xã Phụng Thượng", "Xã Sen Phương", "Xã Tam Hiệp", "Xã Tam Thuấn", "Xã Thanh Đa", "Xã Thọ Lộc", "Xã Thượng Cốc", "Xã Tích Giang", "Xã Trạch Mỹ Lộc", "Xã Vân Hà", "Xã Vân Nam", "Xã Vân Phúc", "Xã Võng Xuyên", "Xã Xuân Đình"    
    },
    "Huyện Ba Vì": {
        "Thị Trấn Tây Đằng", "Xã Ba Trại", "Xã Ba Vì", "Xã Cẩm Lĩnh", "Xã Cam Thượng", "Xã Châu Sơn", "Xã Chu Minh", "Xã Cổ Đô", "Xã Đông Quang", "Xã Đồng Thái", "Xã Khánh Thượng", "Xã Minh Châu", "Xã Minh Quang", "Xã Phong Vân", "Xã Phú Châu", "Xã Phú Cường", "Xã Phú Đông", "Xã Phú Phương", "Xã Phú Sơn", "Xã Sơn Đà", "Xã Tản Hồng", "Xã Tản Lĩnh", "Xã Thái Hòa", "Xã Thuần Mỹ", "Xã Thụy An", "Xã Tiên Phong", "Xã Tòng Bạt", "Xã Vân Hòa", "Xã Vạn Thắng", "Xã Vật Lại", "Xã Yên Bài"    
    },
    "Huyện Đan Phượng": {
        "Thị Trấn Phùng", "Xã Đan Phượng", "Xã Đồng Tháp", "Xã Hạ Mỗ", "Xã Hồng Hà", "Xã Liên Hà", "Xã Liên Hồng", "Xã Liên Trung", "Xã Phương Đình", "Xã Song Phượng", "Xã Tân Hội", "Xã Tân Lập", "Xã Thọ An", "Xã Thọ Xuân", "Xã Thượng Mỗ", "Xã Trung Châu"    
    },
    "Huyện Quốc Oai": {
        "Thị Trấn Quốc Oai", "Xã Cấn Hữu", "Xã Cộng Hòa", "Xã Đại Thành", "Xã Đồng Quang", "Xã Đông Yên", "Xã Hòa Thạch", "Xã Liệp Tuyết", "Xã Nghĩa Hương", "Xã Ngọc Liệp", "Xã Ngọc Mỹ", "Xã Phú Cát", "Xã Phú Mãn", "Xã Phượng Cách", "Xã Sài Sơn", "Xã Tân Hòa", "Xã Tân Phú", "Xã Thạch Thán", "Xã Tuyết Nghĩa", "Xã Yên Sơn", "Xã Đông Xuân"
    },
    "Huyện Hoài Đức": {
        "Thị Trấn Trạm Trôi", "Xã An Khánh", "Xã An Thượng", "Xã Cát Quế", "Xã Đắc Sở", "Xã Di Trạch", "Xã Đông La", "Xã Đức Giang", "Xã Đức Thượng", "Xã Dương Liễu", "Xã Kim Chung", "Xã La Phù", "Xã Lại Yên", "Xã Minh Khai", "Xã Sơn Đồng", "Xã Song Phương", "Xã Tiền Yên", "Xã Vân Canh", "Xã Vân Côn", "Xã Yên Sở"    
    },
    "Huyện Thạch Thất": {
        "Thị Trấn Liên Quan", "Xã Bình Phú", "Xã Bình Yên", "Xã Cẩm Yên", "Xã Cần Kiệm", "Xã Canh Nậu", "Xã Chàng Sơn", "Xã Đại Đồng", "Xã Dị Nậu", "Xã Đồng Trúc", "Xã Hạ Bằng", "Xã Hương Ngải", "Xã Hữu Bằng", "Xã Kim Quan", "Xã Lại Thượng", "Xã Phú Kim", "Xã Phùng Xá", "Xã Tân Xã", "Xã Thạch Hòa", "Xã Thạch Xá", "Xã Tiến Xuân", "Xã Yên Bình", "Xã Yên Trung"
    },
    "Huyện Thanh Oai": {
        "Thị Trấn Kim Bài", "Xã Bích Hòa", "Xã Bình Minh", "Xã Cao Dương", "Xã Cao Viên", "Xã Cự Khê", "Xã Dân Hòa", "Xã Đỗ Động", "Xã Hồng Dương", "Xã Kim An", "Xã Kim Thư", "Xã Liên Châu", "Xã Mỹ Hưng", "Xã Phương Trung", "Xã Tam Hưng", "Xã Tân Ước", "Xã Thanh Cao", "Xã Thanh Mai", "Xã Thanh Thùy", "Xã Thanh Văn", "Xã Xuân Dương"    
    },
    "Huyện Chương Mỹ": {
        "Thị Trấn Chúc Sơn", "Thị Trấn Xuân Mai", "Xã Đại Yên", "Xã Đông Phương Yên", "Xã Đông Sơn", "Xã Đồng Lạc", "Xã Đồng Phú", "Xã Hòa Chính", "Xã Hoàng Diệu", "Xã Hoàng Văn Thụ", "Xã Hồng Phong", "Xã Hợp Đồng", "Xã Hữu Văn", "Xã Lam Điền", "Xã Mỹ Lương", "Xã Nam Phương Tiến", "Xã Ngọc Hòa", "Xã Phú Nam An", "Xã Phú Nghĩa", "Xã Phụng Châu", "Xã Quảng Bị", "Xã Tân Tiến", "Xã Tiên Phương", "Xã Tốt Động", "Xã Thanh Bình", "Xã Thủy Xuân Tiên", "Xã Thụy Hương", "Xã Thượng Vực", "Xã Trần Phú", "Xã Trung Hòa", "Xã Trường Yên", "Xã Văn Võ"    
    },
    "Huyện Thường Tín": {
        "Thị Trấn Thường Tín", "Xã Chương Dương", "Xã Dũng Tiến", "Xã Duyên Thái", "Xã Hà Hồi", "Xã Hiền Giang", "Xã Hòa Bình", "Xã Khánh Hà", "Xã Hồng Vân", "Xã Lê Lợi", "Xã Liên Phương", "Xã Minh Cường", "Xã Nghiêm Xuyên", "Xã Nguyễn Trãi", "Xã Nhị Khê", "Xã Ninh Sở", "Xã Quất Động", "Xã Tân Minh", "Xã Thắng Lợi", "Xã Thống Nhất", "Xã Thư Phú", "Xã Tiền Phong", "Xã Tô Hiệu", "Xã Tự Nhiên", "Xã Vạn Điểm", "Xã Văn Bình", "Xã Văn Phú", "Xã Văn Tự", "Xã Vân Tảo"
    },
    "Huyện Mỹ Đức": {
        "Thị Trấn Đại Nghĩa", "Xã An Mỹ", "Xã An Phú", "Xã An Tiến", "Xã Bột Xuyên", "Xã Đại Hưng", "Xã Đốc Tín", "Xã Đồng Tâm", "Xã Hồng Sơn", "Xã Hợp Thanh", "Xã Hợp Tiến", "Xã Hùng Tiến", "Xã Hương Sơn", "Xã Lê Thanh", "Xã Mỹ Thành", "Xã Phù Lưu Tế", "Xã Phúc Lâm", "Xã Phùng Xá", "Xã Thượng Lâm", "Xã Tuy Lai", "Xã Vạn Kim", "Xã Xuy Xá"    
    },
    "Huyện Phú Xuyên": {
        "Thị Trấn Phú Xuyên", "Thị Trấn Phú Minh", "Xã Bạch Hạ", "Xã Châu Can", "Xã Chuyên Mỹ", "Xã Đại Thắng", "Xã Đại Xuyên", "Xã Hoàng Long", "Xã Hồng Minh", "Xã Hồng Thái", "Xã Khai Thái", "Xã Minh Tân", "Xã Nam Phong", "Xã Nam Tiến", "Xã Nam Triều", "Xã Phú Túc", "Xã Phú Yên", "Xã Phúc Tiến", "Xã Phượng Dực", "Xã Quang Lãng", "Xã Quang Trung", "Xã Sơn Hà", "Xã Tân Dân", "Xã Tri Thủy", "Xã Tri Trung", "Xã Văn Hoàng", "Xã Vân Từ"    
    },
    "Huyện Ứng Hòa": {
        "Thị Trấn Vân Đình", "Xã Cao Thành", "Xã Đại Cường", "Xã Đại Hùng", "Xã Đội Bình", "Xã Đông Lỗ", "Xã Đồng Tiến", "Xã Đồng Tân", "Xã Hoa Sơn", "Xã Hòa Lâm", "Xã Hòa Nam", "Xã Hòa Phú", "Xã Hòa Xá", "Xã Hồng Quang", "Xã Kim Đường", "Xã Liên Bạt", "Xã Lưu Hoàng", "Xã Minh Đức", "Xã Phù Lưu", "Xã Phương Tú", "Xã Quảng Phú Cầu", "Xã Sơn Công", "Xã Tảo Dương Văn", "Xã Trầm Lộng", "Xã Trung Tú", "Xã Trường Thịnh", "Xã Vạn Thái", "Xã Viên An", "Xã Viên Nội"
    }
}
projects = ['C14 - Bộ Công An', 'Khu Ngoại Giao Đoàn', 'Sunshine City', 'Sky Park Residence', 'The Pavilion - Vinhomes Ocean Park', 'Nhà ở xã hội Rice City Tây Nam Linh Đàm', 'Ecolife Capitol', 'Mulberry Lane', 'HH1 Linh Đàm',
            'Times Tower - HACC1 Complex Building', 'Eco Green City', 'Kim Văn Kim Lũ', 'Bea Sky', 'The Sparks', 'Trung Hòa Nhân Chính', '335 Cầu Giấy', 'Gelexia Riverside', 'Masteri West Heights', 'Vinhomes Ocean Park Gia Lâm', 
            'Hà Nội Melody Residences', 'Phú Thịnh Green Park', 'The Nine Tower', 'Tây Hà Tower', 'Eurowindow Multi Complex', 'Times City', 'Xuân Mai Complex', 'Khu đô thị Văn Khê', 'Eco Lake View', 'Chung cư The Wisteria', 'Chung cư 30T Nam Trung Yên', 
            'The Pride', 'The London - Vinhomes Ocean Park', 'Mon City - Hải Đăng City', 'Tháp Doanh Nhân - Boss Tower', 'HH2 Linh Đàm', 'KĐTM Đại Kim - Định Công', 'Mipec City View', '90 Nguyễn Tuân', 'Đường Phạm Hùng', 'Goldmark City', 'Vinhomes Ocean Park', 
            'Udic Westlake', 'Samsora Premier', 'Sunshine Golden River', 'The Two Residence', 'The Nine', 'King Palace', 'Thăng Long Garden 250 Minh Khai', 'Khu đô thị Nam Trung Yên', 'Đồng Phát Park View Tower', 'Richland Southern', 'Mỹ Đình Pearl', 'Vinhomes Green Bay Mễ Trì', 
            'Hà Nội Paragon', 'The Park Home', 'Golden Park Tower', 'Imperia Smart City', 'Newtatco Complex', 'Vinhomes Smart City', 'Văn Khê', 'Thăng Long Number One', 'Helios Tower 75 Tam Trinh', 'Văn Phú Victoria', 'An Bình Plaza', 'Hòa Bình Green City', 
            'Imperia Sola Park', 'Pandora 53 Triều Khúc', 'CT2 Xuân Phương', 'Sunshine Riverside', 'Việt Đức Complex', 'The Emerald CT8 Mỹ Đình', 'Golden Palace', 'The Sapphire-Vinhomes Smart City', 'The Sakura - Vinhomes Smart City', 'HH3 Linh Đàm', 'Tân Tây Đô', 
            'Chung cư The Senique Hanoi', 'Mandarin Garden 2', 'Pháp Vân Tứ Hiệp', "Vinhomes D' Capitale Trần Duy Hưng", 'Mỹ Đình Plaza 2', 'Hồ Gươm Plaza', 'The Garden Hills - 99 Trần Bình', 'Indochina Plaza', 'Roman Plaza', 'Khu Nhà ở KD Dịch Vọng', 'Diamond Flower Tower', 
            'C3 Lê Văn Lương (Golden Palace)', 'Bamboo Airways Tower', 'Handi Resco Lê văn Lương', 'Việt Hưng', 'Licogi 13 Tower', 'KĐT Xa La', 'Dreamland Bonanza', 'The Zenpark', 'Đường Nguyễn Trãi', 'X2 Đại Kim', 'Khu đô thị Mỹ Đình I', 'Mipec Rubik 360', 'IA20 Ciputra', 'Đường Trần Phú', 
            'VP3 Linh Đàm', 'A10-A14 Nam Trung Yên', 'Masteri Water Front', 'Đại Thanh', 'Thành Phố Giao Lưu', 'Eurowindow River Park', 'CTM Building 139 Cầu Giấy', 'D’. Le Roi Soleil', 'Masteri Waterfront', 'Cầu Giấy Center Point', 'An Bình City', 'The Two Residence - Gamuda Garden', 'Royal City', 
            'Nam Xa la', 'Chung cư Ban cơ yếu Chính phủ', 'Sunshine Garden', 'Tòa nhà 197 Trần Phú', 'BRG Diamond Residence', 'HH2 Bắc Hà', 'Licogi 12', 'Vinhomes Skylake', 'Berriver Jardin', 'Cienco1 Hoàng Đạo Thúy', 'Riverside Garden', 'MHD Trung Văn', 'Vimeco II - Nguyễn Chánh', 'Green Park CT15 Việt Hưng', 
            'KĐT Linh Đàm', '173 Xuân Thủy', 'Times City - Park Hill', 'Mandarin Garden', 'Nam Đô Complex', 'Iris Garden', 'Vinhomes Gardenia', 'Đường Giải Phóng', 'Dream Town', 'Đền Lừ II', 'Đông Đô', 'Rose Town 79 Ngọc Hồi', 'Đường Thanh Nhàn', 'BID Residence', 'Rivera Park Hà Nội', 'Usilk City', 'Chung cư Đại Thanh', 
            'Green Stars', 'Chung cư Viện 103', 'Golden Land', 'Sails Tower', 'The Beverly - Vinhomes Ocean Park', 'Phương Đông Green Park', 'Tecco Skyville', 'KĐT Trung Yên', 'Bắc Hà Tower C37 Bộ Công An', 'Vinhomes Symphony Riverside', 'The Golden Palm Lê Văn Lương', 'Park Kiara', 'Sunrise Building 3', 
            'Osaka Complex', 'Hoàng Thành Pearl', 'VOV Mễ Trì', 'Rose Town', 'D’. El Dorado', 'Phenikaa Hòa Lạc', 'Tecco Garden', 'The Zei Mỹ Đình', 'Phú Gia Residence', 'TĐC Hoàng Cầu', 'Westa Hà Đông', 'Yên Hòa Condominium', 'Khu đô thị mới Dịch Vọng', 'Bắc Hà Fodacon', 'KĐT Hạ Đình', 'Nhà ở xã hội Đồng Mô Đại Kim', 
            'Tòa nhà FLC Twin Towers', 'Khu đô thị Vân Canh', 'Central Field Trung Kính', 'The Golden Armor', 'Xuân Mai Tower - CT2 Tô Hiệu', 'KĐT Văn Quán', 'N01-T6 Ngoại Giao Đoàn', 'AQH Riverside', 'Tây Nam Hồ Linh Đàm', 'KĐTM Dương Nội', 'Rice City Sông Hồng', 'Hateco Hoàng Mai', 'GoldSeason', 'N03-T3&T4 Ngoại Giao Đoàn', 
            'Thăng Long Capital', 'The Canopy Residences - Vinhomes Smart City', 'Mỹ Đình I', 'B1-B2 Tây Nam Linh Đàm', 'Tràng An Complex', 'Hanhomes Blue Star', 'Lumiere EverGreen', 'Ha Do Park View', 'Lilama 52 Lĩnh Nam', "Vinhomes D'Capitale", 'Mỹ Đình II', 'Hateco Green City', 'Bình Minh Garden', 'Viha Complex', 
            'CT5-CT6 Lê Đức Thọ', 'PVV-Vinapharm 60B Nguyễn Huy Tưởng', 'ĐTM Dịch Vọng', 'Chelsea Park - Khu đô thị mới Yên Hòa', 'HPC Landmark 105', 'Capital Elite', 'Tháp doanh nhân Tower', 'Khai Sơn Town', 'HTT Tower 3', 'Le Grand Jardin Sài Đồng', 'Đường Quốc Lộ 5', 'Lạc Hồng Westlake', '54 Hạ Đình', 'Eco City Việt Hưng', 
            'Nhà ở xã hội Bộ Tư lệnh Tăng Thiết Giáp', 'The Zen Residence', 'The Victoria - Vinhomes Smart City', 'N01-T3 Ngoại Giao Đoàn', 'Khu đô thị Mỹ Đình Sông Đà', 'Hinode City', 'The Miami', 'Ecolife Tây Hồ', 'Thanh Hà Mường Thanh', 'Ruby City', 'FLC Green Apartment', 'Hapulico Complex', 'Anland 2', 'Sài Đồng', 'Booyoung Vina', 
            'The Terra An Hưng', 'TNR The Nosta', 'Scitech Tower', 'Bohemia Residence', 'D’. El Dorado II', 'Hateco Apollo Xuân Phương', 'Seasons Avenue', 'The Sun Mễ Trì', 'Imperia Garden', 'Vinata Tower', 'Phường Mỗ Lao', 'HH4 Linh Đàm', 'N01-T5 Ngoại Giao Đoàn', 'Imperial Plaza', 'Green Park Tower', 'Tòa Tháp Thiên Niên Kỷ', 'Mỹ Đình', 
            'FLC Complex Phạm Hùng', 'SDU - 143 Trần Phú', 'CT2 Trung Văn Viettel Hancic', 'Park Hill Premium - Times City', 'Ciputra Hà Nội', 'Vimeco I - Phạm Hùng', 'TTTM TSQ', 'Gemek Tower', 'An Lạc Phùng Khoang', 'Khu đô thị mới Xa La', 'Phương Đông Green Home (CT8C Việt Hưng)', 'Golden West', 'Ruby City Long Biên', 'Stellar Garden', 
            'Hà Thành Plaza', 'Lancaster Hà Nội', 'M3 - M4 Nguyễn Chí Thanh', '30 Phạm Văn Đồng', '282 Lĩnh Nam', 'Trung Yên I', 'The Tonkin - Vinhomes Smart City', 'Khai Sơn City', 'Khu đô thị Trung Văn - Vinaconex 3', 'KĐT Đại Kim', 'Xuân Mai Riverside', 'Star Tower 283 Khương Trung', 'Five Star Kim Giang', 'Khu đô thị Việt Hưng', 
            'FLC Star Tower', 'VOV Mễ Trì Plaza', 'Tabudec Plaza', 'Sunshine Center', 'MD Complex Mỹ Đình', 'Phố Khâm Thiên', '89 Phùng Hưng', 'Vinaconex 21', 'Khu đô thị mới Dương Nội', 'Thống Nhất Complex', 'KĐT Định Công', '249A Thụy Khuê', 'Mỹ Gia 1 Phú Mỹ Hưng', 'Vinpearl Đà Nẵng Resort and Villas', '6 Đội Nhân', 'T&T Riverview', 
            'HUD3 Nguyễn Đức Cảnh', 'N04B Ngoại Giao Đoàn', 'Đường Mỗ Lao', 'SDU 143 Trần Phú', 'Keangnam', 'Khu đô thị mới Cầu Giấy', 'Hồng Hà Eco City', 'Handico Complex', 'Hyundai Hillstate', 'Hong Kong Tower', 'CT3 Tây Nam Linh Đàm', 'Ecohome 1', '57 Vũ Trọng Phụng', 'Hei Tower Điện Lực', 'Khu đô thị Trung Hòa - Nhân Chính', 'Vinhomes Cổ Loa', 
            'Anland LakeView', 'Eco Dream', '789 Bộ Quốc Phòng - 147 Hoàng Quốc Việt', 'Feliz Homes', 'Green Park Trần Thủ Độ', 'Golden Field Mỹ Đình', 'The Zurich', 'Sông Hồng Park View', 'NO-08 Giang Biên', 'B.I.G Tower', 'Hà Nội Center Point', 'Golden Westlake', 'Grand Sunlake', 'Summit Building', 'Đường Định Công', 'Sky Central', 
            'Khu đô thị Vinhomes SkyLake', 'Đường Hoàng Mai', 'The One Residence - Gamuda Garden', 'Xuân Mai Sparks Tower', 'NHS Phương Canh Residence', 'The Golden An Khánh', 'Chung cư An Sinh', 'Nhà ở cho CBCS Bộ Công an', 'Intracom1', 'Đường Linh Đường', 'Sunshine Palace', 'The Link 345-CT1', 'Anland Premium', 'Dolphin Plaza', 'Đường Nhân Mỹ', 
            'Đường Mễ Trì Hạ', 'Tây Hồ Residence', 'The Pavilion', 'Nhà ở xã hội @Home', 'CT3 C’Land Lê Đức Thọ', 'TSG Lotus Sài Đồng', 'Lumi Hanoi', 'Khu đô thị mới Pháp Vân - Tứ Hiệp', 'Khu đô thị mới Linh Đàm', 'N03-T7 Ngoại Giao Đoàn', 'Lucky House', 'Tòa nhà Sông Hồng Park View', 'Park View City', 'Berriver Long Biên', 'Yên Hòa', 'Hancorp Plaza', 
            'Xuân Phương Residence', 'Thăng Long Victory', 'Sông Đà 7', 'Khu đô thị mới Đại Kim', '6th Element', 'Học Viện Quốc Phòng', 'Đường Trần Thái Tông', 'Vinaconex 7 - 34 Cầu Diễn', 'Phường Trần Phú', 'Legend Tower 109 Nguyễn Tuân', 'Chung cư CT3 Nghĩa Đô', 'Đường Trung Phụng', 'Resco Cổ Nhuế', '96 Định Công', 'Khu đô thị Kim Văn Kim Lũ', 'KĐT Mễ Trì Hạ', 
            'Packexim', 'HC Golden City', 'HTT Tower', 'CT36 - Dream Home', 'VP5 Linh Đàm', 'Khu nhà ở Bắc Hà', 'The One Residence', 'Đường Nguyễn Cơ Thạch', 'Booyoung', 'Vinhomes Nguyễn Chí Thanh', 'GoldSilk Complex', 'Anland Complex', 'The Vesta', 'Momota Nguyễn Đức Cảnh', 'Hateco Apollo', 'Vinhomes West Point', 'Florence Mỹ Đình', 'Ecohome 3', 'Khu đô thị Thanh Hà Cienco 5', 
            'Đường Đại lộ Thăng Long', 'Đường Nguyễn Hữu Thọ', 'Hoa Dao Hotel', 'Packexim 2 Tây Hồ', 'CT2 Yên Nghĩa', 'Vinaconex 1', 'Vinhomes Symphony', 'The Zurich - Vinhomes Ocean Park', 'Northern Diamond', 'Đường 70', '25 Tân Mai', '671 Hoàng Hoa Thám', 'Lancaster Luminaire', 'KĐT Vĩnh Hoàng', 'CT Number One', 'Kosmo Tây Hồ', 'Căn hộ Thông tấn xã', 'Phố Đại La', 'xpHOMES', 
            'Tứ Hiệp Plaza', 'Khu đô thị Vinhomes Gardenia', 'Bình Vượng 200 Quang Trung', 'Skyline West Lake', 'Hạ Đình Tower', 'Đường Vĩnh Phúc', 'N04A Ngoại Giao Đoàn', 'Moonlight 1 - An Lạc Green Symphony', 'Him Lam Thạch Bàn 2', 'Mễ Trì Thượng', 'Khu đô thị Xuân Phương', 'Home City - Trung Kính Complex', 'Green House', 'Housinco Premium', 'Chung cư 789 Xuân Đỉnh', 'Đường Lương Ngọc Quyến', 
            'Ruby City CT3 Phúc Lợi', 'Hope Residence', 'Lumi Elite', 'AZ Sky', 'N01-T1 Ngoại Giao Đoàn', 'Helios Tower', 'Đường Hàm Nghi', 'Grand SunLake Văn Quán', 'Sudico Mỹ Đình', 'Imperia Sky Garden', 'Đường Nguyễn Công Trứ', 'Sakura Tower', 'Rice City Linh Đàm', 'Lidaco-Vinaconex 7', 'CT4 Vimeco II', 'Hinode Royal Park', 'Gold Tower', 'Đặng Xá 1', 'ICID Complex', 'An Lạc - Mỹ Đình', 
            'Đường Giáp Nhị', 'Tây Nam Đại học Thương Mại', 'Chelsea Residences', 'Epic Tower', 'Newhouse Xa La', 'Viễn Đông Star', 'VP2 Linh Đàm', 'Heritage West Lake', 'đường Mỹ Đình', 'Khu đô thị Eurowindow River Park', 'The K Park', 'N02-T2 Ngoại Giao Đoàn', 'The Charm An Hưng', 'KĐT mới Cầu Giấy', 'Phường Mỹ Đình 1', 'D11 Sunrise Building', '16B Nguyễn Thái Học', 'Chung cư The Legacy', 
            'An Lạc Green Symphony', 'Phường Giang Biên', 'Đường Nguyễn Văn Lộc', 'FLC Garden City', 'Khai Sơn Hill', 'Sapphire Palace', 'Phường Mai Động', 'Green Pearl 378 Minh Khai', 'The Matrix One', 'Hà Nội Homeland', 'Thủy Lợi Tower', 'M5 Nguyễn Chí Thanh', 'Hemisco Xa La', 'Làng Việt Kiều Châu Âu Euroland', 'Sun Square', 'Khu đô thị mới Yên Hòa', 'KĐT Cổ Nhuế', 'Hà Đô Park View', 'Nghĩa Đô', 
            'Anland Lakeview', 'Ecohome Phúc Lợi', 'Sky City Towers-88 Láng Hạ', 'Samsora Premier 105', 'Tân Việt Tower', 'K35 Tân Mai', 'Hồng Hà Tower', 'PCC1 Complex', 'Discovery Complex', 'Smile Building', 'Geleximco Southern Star', 'Nam Trung Yên', 'Gemek Premium', 'Phường Thịnh Liệt', 'Chung cư Ruby City CT3', 'CT2 Trung Văn - Vinaconex 3', 'Chung cư C1 Thành Công', 'Luxury Park Views', 'Đường Đại Cồ Việt', 
            'Comatce Tower', 'Phường Dịch Vọng', 'Mailand Hanoi City', 'New Skyline', 'Green Diamond 93 Láng Hạ', 'HP Landmark Tower', 'Phố Trần Quý Kiên', 'Valencia Garden', 'Star Tower', 'The Manor', 'D’. Le Pont D’or Hoàng Cầu', 'Oriental Westlake', 'Sunrise Garden (Bình Minh Garden)', 'KĐTM Cầu Bươu', '93 Lò Đúc - Kinh Đô Tower', 'The Sparks Dương Nội', 'Kiến Hưng Luxury', 'Harmony Square', 'PHC Complex 158 Nguyễn Sơn', 
            'Ao Sào', 'QMS Tower', '310 Minh Khai', 'Khu đô thị mới Văn Phú', 'Tây Hồ River View', 'Đường Tố Hữu', 'The Garden Hills', 'Athena Complex', 'KĐT Trung Văn - Hancic', 'Phố Lò Đúc', 'Happy Star Tower', 'Khu phức hợp cao tầng Mỹ Đình', 'Phố Lê Thanh Nghị', 'Liền kề 622 Minh Khai', 'Khu đô thị mới Đại Thanh', 'C14 Bộ Quốc Phòng', 'Phú Mỹ', 'AZ Lâm Viên Complex', 'Chung cư Bộ Tổng Tham Mưu', 'The Golden An Khánh 32T', 
            'Khu đô thị Vinhomes Times City', 'Đường Minh Khai', 'FLC Landmark Tower', 'Ecohome 2', 'The Artemis', 'N04 Trần Duy Hưng', 'Hà Đông Park View', '789 Bộ Tổng Tham Mưu - BQP', 'Phùng Khoang', 'C37 Bộ Công An - Bắc Hà Tower', '101 Láng Hạ', 'CT3 Cổ Nhuế', 'THT New City', 'New Horizon City - 87 Lĩnh Nam', 'Unimax Twin Tower', 'Khu đô thị Đại Kim - Định Công', 'Chung cư 122 Vĩnh Tuy', 'La Casta Văn Phú', 'Phường Ngọc Thụy', 
            'VP6 Linh Đàm', 'Mipec Riverside', 'Thanh Xuân Complex', 'Nàng Hương', 'The Sakura', 'Kiến Hưng', 'Tổng cục 5 Bộ Công An', 'Khu đô thị Mễ Trì Hạ', 'Tháp đôi Kepler Land (TSQ Mỗ Lao)', 'Home City', 'T&T Tower', '44 Triều Khúc', 'Park View Residence Dương Nội', 'Đường Trương Định', 'N01-T2 Ngoại Giao Đoàn', 'PCC1 Triều Khúc', 'CT3 Cổ Nhuế', 'Sông Đà Hà Đông Tower', 'CT1 Yên Nghĩa', 'Sunshine Green Iconic', 
            'Hoàng Ngân Plaza', '187 Tây Sơn', '25 Vũ Ngọc Phan', 'Nhà ở xã hội Kiến Hưng - Lucky House', 'Phường Thổ Quan', 'CT2 Viettel Trung Văn', 'CT15 Việt Hưng Green Park', 'PentStudio', 'Diamond Goldmark City', 'Phường Hoàng Liệt', 'Phố Doãn Kế Thiện', 'Khu nhà ở Hưng Thịnh', 'Chung cư 60 Hoàng Quốc Việt', 'Nam La Khê', 'CT1 Thạch Bàn', 'Discovery Central', 'Phường Yên Nghĩa', 'Lumi Prestige', 'X2 Mỹ Đình', 
            'KĐT Tây Nam Kim Giang', 'Fafilm - VNT Tower', 'Vimeco Hoàng Minh Giám', 'An Lạc Mỹ Đình', 'B4 và B14 Kim Liên', 'Phương Đông Green Home', 'Mỹ Sơn Tower', '151 Hoàng Quốc Việt', 'Chung cư Xuân La', 'Nam Xa La', 'Đường Xa La', 'Đường Hoàng Quốc Việt', 'CT3 Yên Nghĩa', 'Đường Nguyễn Duy Trinh', 'Chung cư Thông Tấn Xuân Phương', 'Amber Riverside', 'Lộc Ninh Singashine', 'Moonlight I', 'N01-T7 Ngoại Giao Đoàn', 
            'Phố Trần Tử Bình', 'Khu nhà ở Bộ tư lệnh Thủ đô Hà Nội', 'Sài Đồng Lake View', 'Phố Chùa Bộc', 'Tây Mỗ', 'Khu đô thị Kim Chung - Di Trạch', 'Đường Vũ Trọng Khánh', 'Phường Phúc La', 'Vinacomin Tower', 'Đường Nguyễn Hoàng', 'CT6 Constrexim Yên Hòa', 'Chung cư Han Jardin', 'Chung cư 24 Nguyễn Khuyến', 'Làng Quốc tế Thăng Long', 'Ngô Thì Nhậm', '113 Trung Kính', 'Trương Định Complex', 'B4 - B14 Kim Liên', 'Hateco Laroma', 
            'Trinity Tower', 'Pacific Place', 'Phường Trung Văn', 'Capital Garden 102 Trường Chinh Kinh Đô', 'D22 Bộ Tư Lệnh Biên Phòng', 'KĐT Trung Văn - Vinaconex 3', 'GoldSeason 47 Nguyễn Tuân', 'Tòa nhà N01-T8', 'The Garden', 'Watermark', 'Đường Đình Thôn', 'Mỗ Lao', 'CT36 Dream Home', 'Vườn Xuân - 71 Nguyễn Chí Thanh', 'Spring Home', 'Làng Quốc Tế Thăng Long', 'Đường Nguyễn Quý Đức', 'Lotus Lake View', 'Bình Vượng Tower', 'Nhà ở xã hội EcoHome 2', 
            '108 Nguyễn Trãi', 'Đường Phan Đình Giót', 'KĐT Văn Phú', 'Riverside Tower 79 Thanh Đàm', 'The Golden Palm', 'Times City Park Hill', 'Gamuda City', 'KĐT Tây Hồ Tây - Starlake Hà Nội', 'Khu đô thị Tân Tây Đô', 'Indochina Plaza Hanoi (IPH)', '120 Hoàng Quốc Việt BQP', 'Khu đô thị Splendora An Khánh', 'N01-T8 Ngoại Giao Đoàn', 'N03-T6 Ngoại Giao Đoàn', 'Đường Trần Đăng Ninh', 'Thanh Bình Garden', 'Phố Lương Định Của', 'Khu đô thị ParkCity Hà Nội', 
            'Chung cư 622 Minh Khai', 'Aeon Mall Long Biên', 'N01-D17 Duy Tân', 'Đường Ngọc Lâm', 'N03-T2 Ngoại Giao Đoàn', 'An Bình Tower', 'VP4 Linh Đàm', 'Tecco Diamond Thanh Trì', 'Khu đô thị Vinhomes Royal City', 'Han Jardin', 'Artex Building 172 Ngọc Khánh', 'NOXH Đồng Mô', 'VC7 Housing Complex - 136 Hồ Tùng Mậu', 'Khu nhà ở xã hội 622 Minh Khai', 'Hacinco Complex (Hà Nội Center Point)', 'Phường Mễ Trì', 'MBLand Central Field (Central Point Trung Kính)', 
            'Intracom 1 Trung Văn', 'Đường Đền Lừ', 'The Legend Tower', 'Xuân Phương Tasco', 'New Horizon City', 'Intracom 2 Cầu Diễn', 'Green House Việt Hưng', 'Đường Dương Đình Nghệ', 'Khu đô thị mới Cổ Nhuế', 'FLC Premier Parc Đại Mỗ', 'Trung tâm thương mại TSQ', 'Phố 8/3', 'Đường Giang Văn Minh', '27 Huỳnh Thúc Kháng', 'Nhà ở xã hội Hưng Thịnh', 'IEC Residences Tứ Hiệp', 'Quận Hoàng Mai', 'Khu đô thị mới Văn Quán', 'Parkview Residence', 'La Casta Tower Văn Phú', 
            'One 18 Ngọc Lâm', 'Gamuda Gardens', 'Thăng Long Green City', 'Phố Xã Đàn', 'An Lạc Tower', 'Nhà ở xã hội Bộ công an Cổ Nhuế 2', 'Chung cư Đông Đô', 'Chung cư 536A Minh Khai', 'Phường Nhân Chính', 'Rainbow Linh Đàm', 'Khu phức hợp Imperia Garden', 'Phường Định Công', 'Đường Nguyễn Chánh', 'Đường Nguyễn Đức Cảnh', 'The Link Ciputra', 'Nam An Khánh', 'Đường Lê Văn Lương', 'Liễu Giai Tower', 'Trung Yên Plaza', 'Đặng Xá 2', 'Chung cư Ngô Thì Nhậm', 'Khu đô thị Mỹ Đình II', 
            'C1 C2 Xuân Đỉnh', 'Khu đô thị Trung Văn Hancic', 'Tecco Diamond', 'Tòa nhà 169 Nguyễn Ngọc Vũ', 'Cienco1', 'đường Tây Mỗ', 'Viện bỏng Lê Hữu Trác', 'Mỹ Đình Plaza', 'Núi Trúc Square', 'Chung cư CT4 Yên Nghĩa', 'Nhà ở xã hội NO1 Hạ Đình - UDIC Eco Tower', 'VP7 Linh Đàm', 'Hanoi Melody Residences', 'Five Star Garden', 'Thịnh Liệt', 'Đường Thanh Bình', 'Watermark Tây Hồ', 'CTM 299 Cầu Giấy', 'SME Hoàng Gia', 'Phường Xuân Tảo', 'Petrowaco 97 Láng Hạ', '130 Nguyễn Đức Cảnh', 
            'Yên Hòa Thăng Long', 'N01-T4 Ngoại Giao Đoàn', 'Đường Lương Sử A', 'Chung cư 345 Đội Cấn', 'Đền Lừ I', 'Starcity Lê Văn Lương', 'Đường Lưu Hữu Phước', 'New House Xa La', 'Quận Hai Bà Trưng', 'Five Star Mỹ Đình', 'N03-T1 Ngoại Giao Đoàn', 'Vườn Đào', 'Tổng cục 5 Tân Triều', 'Sky Light', 'Phố Chùa Láng', '16 Liễu Giai', 'Lancaster Núi Trúc', 'Hòa Bình Green Apartment', 'N03-T5 Ngoại Giao Đoàn', 'Đường Kim Ngưu', 'Đường Trần Hữu Dực', 'N05', 'Sun Grand City', 'Đường Tân Mai', 
            '536A Minh Khai', 'Jade Square', 'Đường Nguyễn Chí Thanh', 'Tòa nhà Hei Tower', 'Đường Nghiêm Xuân Yêm', 'Lilama 124 Minh Khai', 'Phường Yên Hòa', 'South Tower Hoàng Liệt', 'Khu đô thị Sài Đồng', 'N02-T3', 'CT1 Trung Văn - Vinaconex 3', 'Khu đô thị Hồng Hà Eco City', 'MIPEC Towers', 'Hà Nội Aqua Central', 'T&T DC Complex', 'Phố Trần Quốc Hoàn', 'Đường Bùi Xương Trạch', 'MHDI X2 Đại Kim', 'Phường Nghĩa Tân', 'FLC Complex', 'N03-T8 Ngoại Giao Đoàn', 'Đường Cầu Bươu', 'The Eden Rose', 
            'Đường Lý Nam Đế', 'Golden Palace Mễ Trì', 'Thành Công Tower', 'Phố Nguyễn Ngọc Vũ', 'Phường Dịch Vọng Hậu', 'Đông Nam Trần Duy Hưng', 'Thang Long Number One', 'Constrexim Complex', 'Momota', 'KĐT La Khê', 'Phường Phú Lương', 'Đường Ngọc Hồi', 'HUD Me Linh Central', '95 Cầu Giấy', 'Thông Tấn Xã Việt Nam', 'The Gloria by Silk Path']

district = st.sidebar.selectbox("Quận/Huyện", list(location_data.keys()))
sub_district = st.sidebar.selectbox("Phường/Xã", sorted(location_data[district]))
du_an = st.sidebar.selectbox("Dự án", projects)
acreage_value = st.sidebar.number_input("Diện tích (m2)", min_value=0.0, value=50.0)
huong = st.sidebar.selectbox("Hướng", [None, "Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"])
phap_ly = st.sidebar.selectbox("Pháp lý", [None, "Giấy chứng nhận quyền sở hữu đất", "Giấy tờ hợp lệ"])
no_bed = st.sidebar.number_input("Số phòng ngủ", min_value=-1, value=-1)
if no_bed == -1:  
    no_bed = None
no_bathroom = st.sidebar.number_input("Số phòng tắm", min_value=-1, value=-1)
if no_bathroom == -1:  
    no_bed = None
so_lau = st.sidebar.number_input("Số lầu", min_value=-1, value=-1)
if so_lau == -1:  
    so_lau = None

location_str = f"{district} - {sub_district}"

if st.sidebar.button("🏡 Predict Price"):
    data = {
        'duAn': [du_an],
        'huong': [huong],
        'phapLy': [phap_ly],
        'noBed': [no_bed],
        'soLau': [so_lau],
        'Precinct': [sub_district],
        'District': [district],
        'acreage_value': [acreage_value],
        'noBathroom': [no_bathroom]
    }
    df = pd.DataFrame(data)
    input_params= preprocessor.transform(df)    
    predicted_price_1 = linear_regression_model.predict(input_params)[0]
    predicted_price_2 = svr_model.predict(input_params)[0]
    predicted_price_3 = xgboost_model.predict(input_params)[0]
    
    st.success(f"💰 Predicted Prices for {location_str}:")
    st.write(f"Linear Regression Model: {predicted_price_1:,.2f} billion VND")
    st.write(f"SVR Model: {predicted_price_2:,.2f} billion VND")
    st.write(f"XGBoost Model: {predicted_price_3:,.2f} billion VND")
