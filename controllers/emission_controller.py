# Mô tả: Lớp điều khiển chính cho việc dự đoán khí thải CO2
# Lớp này đóng vai trò trung gian giữa mô hình và giao diện người dùng

from models.emission_model import EmissionModel
import pandas as pd
import requests
import os
import logging

# Cấu hình logging để theo dõi quá trình thực thi
logging.basicConfig(
    level=logging.INFO,  # Mức độ log: chỉ hiển thị thông tin từ INFO trở lên
    format='%(asctime)s - %(levelname)s - %(message)s'  # Định dạng log: thời gian - mức độ - nội dung
)
logger = logging.getLogger(__name__)  # Khởi tạo logger cho module này

class EmissionController:
    def __init__(self):
        # Khởi tạo EmissionController với các thuộc tính ban đầu
        self.model = EmissionModel()  # Tạo instance của mô hình dự đoán
        self.trained = False  # Trạng thái huấn luyện của mô hình
        self.avg_emission = None  # Giá trị trung bình của khí thải CO2
        # URL API từ biến môi trường hoặc mặc định là localhost
        self.api_url = os.environ.get('API_URL', 'http://localhost:10000') + "/predict"

    def initialize_model(self, data_path):
        """Khởi tạo và huấn luyện mô hình"""
        logger.info("Khởi tạo mô hình...")
        
        # Thử tải mô hình đã huấn luyện trước
        if self.model.load_model():
            logger.info("Đã tải thành công mô hình đã huấn luyện trước đó")
            self.trained = True
        else:
            logger.info("Không tìm thấy mô hình đã huấn luyện. Đang huấn luyện mô hình mới...")
            
        # Lấy điểm kiểm tra (được tính từ mô hình đã tải hoặc từ quá trình huấn luyện)
        test_score = self.model.train(data_path)
        self.trained = True
        
        # Tính toán giá trị khí thải trung bình
        df = self.model.load_and_preprocess_data(data_path)
        self.avg_emission = df['CO2 Emissions(g/km)'].mean()
        
        logger.info(f"Khởi tạo mô hình hoàn tất. Điểm kiểm tra: {test_score:.3f}")
        return test_score

    def predict_emission(self, features):
        """Dự đoán khí thải sử dụng mô hình cục bộ"""
        if not self.trained:
            raise ValueError("Mô hình cần được huấn luyện trước!")
        
        return self.model.predict(features)

    def predict_emission_api(self, features):
        """Dự đoán khí thải sử dụng API và trả về phản hồi đầy đủ bao gồm thời gian xử lý"""
        try:
            # Gửi yêu cầu đến API
            response = requests.post(self.api_url, json=features)
            response.raise_for_status()
            
            # Trả về dữ liệu phản hồi đầy đủ
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Yêu cầu API thất bại: {str(e)}")

    def get_feature_importance(self):
        """Lấy điểm quan trọng của các đặc trưng"""
        if not self.trained:
            raise ValueError("Mô hình cần được huấn luyện trước!")
        
        return self.model.get_feature_importance()

    def get_average_emission(self):
        """Lấy giá trị khí thải trung bình"""
        return self.avg_emission

    def get_emission_rating(self, emission_value):
        """Lấy xếp hạng khí thải (A đến F)"""
        if emission_value < 100:
            return 'A'  # Phát thải rất thấp
        elif emission_value < 120:
            return 'B'  # Phát thải thấp
        elif emission_value < 140:
            return 'C'  # Phát thải trung bình thấp
        elif emission_value < 160:
            return 'D'  # Phát thải trung bình
        elif emission_value < 180:
            return 'E'  # Phát thải cao
        else:
            return 'F'  # Phát thải rất cao

    def get_eco_tips(self, emission_value):
        """Cung cấp mẹo thân thiện với môi trường dựa trên giá trị khí thải"""
        tips = []
        if emission_value > 160:
            # Mẹo cho phương tiện phát thải cao
            tips.extend([
                "Xem xét chuyển sang phương tiện tiết kiệm nhiên liệu hơn",
                "Bảo dưỡng định kỳ có thể giúp giảm khí thải",
                "Tránh tăng tốc và phanh mạnh"
            ])
        if emission_value > 140:
            # Mẹo cho phương tiện phát thải trung bình
            tips.extend([
                "Kiểm tra áp suất lốp thường xuyên",
                "Loại bỏ trọng lượng dư thừa khỏi phương tiện"
            ])
        # Mẹo chung cho mọi phương tiện
        tips.extend([
            "Sử dụng kỹ thuật lái xe sinh thái",
            "Lên kế hoạch cho các chuyến đi để tránh tắc nghẽn giao thông"
        ])
        return tips 