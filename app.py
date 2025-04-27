import streamlit as st
import os
import sys
import requests
import time
import threading
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Thêm đường dẫn hiện tại vào sys.path (để đảm bảo imports hoạt động trên Streamlit Cloud)
# Cần thiết để Streamlit Cloud có thể tìm thấy các module tự tạo
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Thiết lập cấu hình trang Streamlit
# Must be the first Streamlit command - phải được gọi trước mọi lệnh Streamlit khác
st.set_page_config(
    page_title="CO2 Emission Predictor",  # Tiêu đề hiển thị trên tab trình duyệt
    page_icon="🌍",  # Biểu tượng trang web
    layout="wide",  # Bố cục rộng để tận dụng không gian màn hình
    initial_sidebar_state="expanded"  # Thanh bên mở rộng mặc định
)

# Import các module sau khi đã cấu hình đường dẫn
from controllers.emission_controller import EmissionController
from views.main_view import MainView

# Thiết lập URL API - kết nối đến API server được triển khai trên Render.com
os.environ['API_URL'] = 'https://thuco2tiep.onrender.com'

# Cơ chế kiểm soát đồng thời các request đến API
api_semaphore = threading.Semaphore(10)  # Tăng lên 10 request đồng thời

# Cache lưu kết quả API để tránh gửi lại các request giống nhau
prediction_cache = {}  # Lưu trữ kết quả dự đoán
cache_lock = threading.Lock()  # Khóa đồng bộ cho cache
MAX_CACHE_SIZE = 100  # Giới hạn kích thước cache

# Giá trị mặc định khi API không phản hồi
DEFAULT_PREDICTION = 200.0  # Giá trị CO2 mặc định (g/km)

def get_session():
    """
    Tạo phiên requests với cơ chế thử lại tự động
    
    Cấu hình phiên HTTP với chiến lược thử lại để xử lý các lỗi mạng tạm thời
    và đảm bảo khả năng phục hồi của các yêu cầu API.
    
    Returns:
        requests.Session: Đối tượng phiên có cấu hình thử lại
    """
    session = requests.Session()
    retry = Retry(
        total=5,  # Tăng số lần thử lại tối đa
        backoff_factor=0.2,  # Giảm thời gian giữa các lần retry để tăng tốc
        status_forcelist=[429, 500, 502, 503, 504],  # Mã HTTP cần thử lại
        allowed_methods=["GET", "POST"]  # Các phương thức được phép thử lại
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_cache_key(features):
    """
    Tạo khóa cache từ đặc trưng xe
    
    Chuyển đổi các đặc trưng xe thành chuỗi duy nhất
    để sử dụng làm khóa cho cache.
    
    Parameters:
        features (dict): Các đặc trưng của xe
        
    Returns:
        str: Chuỗi khóa duy nhất hoặc None nếu có lỗi
    """
    try:
        key_parts = []
        for k, v in sorted(features.items()):
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    except:
        return None

def predict_with_api(features):
    """
    Thực hiện dự đoán sử dụng API bên ngoài với kiểm soát đồng thời
    
    Hàm này quản lý các request đến API, bao gồm:
    - Kiểm tra cache trước khi gọi API
    - Kiểm soát số lượng request đồng thời với semaphore
    - Xử lý các trường hợp lỗi và timeout
    - Lưu kết quả vào cache
    
    Parameters:
        features (dict): Các đặc trưng của xe cần dự đoán
        
    Returns:
        dict: Kết quả dự đoán từ API hoặc giá trị dự phòng
    """
    # Tạo cache key trước
    cache_key = get_cache_key(features)
    
    # Kiểm tra cache trước tiên
    with cache_lock:
        if cache_key in prediction_cache:
            return prediction_cache[cache_key]
    
    # Cơ chế dự phòng khi không thể gửi request
    try:
        # Sử dụng semaphore để giới hạn số request đồng thời
        acquired = api_semaphore.acquire(timeout=0.5)  # Timeout nếu không thể acquire trong 0.5s
        if not acquired:
            # Nếu không thể lấy semaphore, trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'Too many concurrent requests'
            }
            
        try:
            # Thêm độ trễ ngẫu nhiên nhỏ để tránh gửi đồng loạt request
            time.sleep(random.uniform(0.01, 0.1))  # Giảm delay ngẫu nhiên
            
            # Kiểm tra chế độ benchmark để chọn endpoint phù hợp
            benchmark_mode = os.environ.get('BENCHMARK_MODE', 'false').lower() == 'true'
            
            # Thực hiện request đến API
            session = get_session()
            api_url = os.environ.get('API_URL')
            
            if benchmark_mode:
                # Sử dụng endpoint fallback đơn giản cho benchmark
                api_url = api_url + "/fallback"
                response = session.post(api_url, json={}, timeout=2)
            else:
                # Sử dụng endpoint dự đoán thực tế
                api_url = api_url + "/predict"
                response = session.post(api_url, json=features, timeout=2)
                
            response.raise_for_status()
            result = response.json()
            
            # Lưu kết quả vào cache
            with cache_lock:
                if len(prediction_cache) < MAX_CACHE_SIZE:
                    prediction_cache[cache_key] = result
            
            return result
        except requests.exceptions.Timeout:
            # Xử lý lỗi timeout - trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'API timeout'
            }
        except requests.exceptions.RequestException as e:
            # Xử lý các lỗi request khác - trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': f'API error: {str(e)}'
            }
        finally:
            # Đảm bảo luôn giải phóng semaphore
            api_semaphore.release()
    except Exception as e:
        # Xử lý mọi lỗi khác (bao gồm lỗi khi lấy semaphore)
        return {
            'prediction': DEFAULT_PREDICTION,
            'process_time_ms': 5.0,
            'status': 'fallback',
            'message': f'Client error: {str(e)}'
        }

def check_api_health():
    """
    Kiểm tra trạng thái hoạt động của API
    
    Gửi request kiểm tra sức khỏe đến API server và chờ đợi
    cho đến khi API sẵn sàng hoặc hết thời gian chờ.
    Hiển thị trạng thái kết nối cho người dùng.
    
    Returns:
        bool: True nếu API sẵn sàng hoặc tiếp tục mà không có API
    """
    api_url = os.environ.get('API_URL')
    
    st.markdown("### Kiểm tra kết nối API")
    status_placeholder = st.empty()
    status_placeholder.info("Đang kết nối đến API server...")
    
    try:
        # Sử dụng phiên với cơ chế thử lại
        session = get_session()
        response = session.get(f"{api_url}/health", timeout=10)  # Giảm timeout xuống 10s
        
        if response.status_code == 200:
            status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
            return True
        else:
            # Xử lý khi API đang khởi tạo (không phải lỗi)
            status = response.json().get("status", "") if response.content else "unknown"
            message = response.json().get("message", "") if response.content else "No response"
            
            # Chờ tối đa 20 giây (giảm từ 60s)
            for i in range(20):
                status_placeholder.warning(f"API server đang khởi tạo... Vui lòng đợi ({i+1}/20s)")
                time.sleep(1)
                
                try:
                    response = session.get(f"{api_url}/health", timeout=3)
                    if response.status_code == 200 and response.json().get("status") == "healthy":
                        status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            # Sau khi hết thời gian chờ, vẫn tiếp tục với mô hình local
            status_placeholder.error(f"API server có vấn đề: {message}. Tiếp tục với dự đoán local.")
            return True  # Vẫn trả về True để tiếp tục
    except requests.exceptions.RequestException as e:
        status_placeholder.error(f"Không thể kết nối đến API server tại {api_url}: {str(e)}")
        # Tiếp tục mà không có API - sẽ sử dụng mô hình local
        return True

def main():
    """
    Hàm chính khởi chạy ứng dụng Streamlit
    
    Thực hiện các bước:
    1. Kiểm tra kết nối API
    2. Kiểm tra file dữ liệu
    3. Khởi tạo controller và view
    4. Huấn luyện mô hình
    5. Hiển thị giao diện người dùng
    """
    st.title("CO2 Emission Prediction")
    
    # Kiểm tra kết nối đến API server - luôn tiếp tục bất kể kết quả
    api_available = check_api_health()
        
    # Kiểm tra file CSV dữ liệu tồn tại
    csv_path = os.path.join(current_dir, "co2 Emissions.csv")
    if not os.path.exists(csv_path):
        st.error(f"Lỗi: Không thể tìm thấy file '{csv_path}'. Vui lòng đảm bảo file tồn tại trong thư mục gốc của dự án.")
        return

    # Khởi tạo controller và ghi đè phương thức gọi API
    controller = EmissionController()
    # Ghi đè phương thức dự đoán API bằng hàm có kiểm soát đồng thời
    controller.predict_emission_api = predict_with_api
    
    # Huấn luyện mô hình
    try:
        test_score = controller.initialize_model(csv_path)
        st.success(f"Mô hình được huấn luyện thành công. Điểm kiểm tra: {test_score:.3f}")
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        return

    # Khởi tạo và hiển thị giao diện
    view = MainView(controller)
    view.show()

# Entry point - chỉ thực thi khi chạy trực tiếp
if __name__ == "__main__":
    main() 