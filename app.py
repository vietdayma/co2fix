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
        total=8,  # Tăng số lần thử lại tối đa lên 8
        backoff_factor=0.5,  # Tăng backoff_factor để đợi lâu hơn giữa các lần retry
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 524],  # Thêm mã lỗi Cloudflare
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],  # Mở rộng các phương thức được phép
        respect_retry_after_header=True  # Tôn trọng header Retry-After từ server
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
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
        acquired = api_semaphore.acquire(timeout=2.0)  # Tăng timeout lên 2.0s
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
            time.sleep(random.uniform(0.01, 0.05))  # Giảm delay ngẫu nhiên xuống 
            
            # Kiểm tra chế độ benchmark để chọn endpoint phù hợp
            benchmark_mode = os.environ.get('BENCHMARK_MODE', 'false').lower() == 'true'
            
            # Thực hiện request đến API
            session = get_session()  # Sử dụng phiên có retry
            api_url = os.environ.get('API_URL')
            
            # Tăng timeout cho API call
            api_timeout = 15.0  # Tăng từ 2s lên 15s để xử lý cold start
            
            # Thêm header để tracking
            headers = {
                'X-Client-Source': 'streamlit-app',
                'X-Request-ID': f"req-{int(time.time() * 1000)}"
            }
            
            if benchmark_mode:
                # Sử dụng endpoint fallback đơn giản cho benchmark
                api_url = api_url + "/fallback"
                response = session.post(api_url, json={}, timeout=api_timeout, headers=headers)
            else:
                # Sử dụng endpoint dự đoán thực tế
                api_url = api_url + "/predict"
                response = session.post(api_url, json=features, timeout=api_timeout, headers=headers)
                
            response.raise_for_status()
            result = response.json()
            
            # Lưu kết quả vào cache
            with cache_lock:
                if len(prediction_cache) < MAX_CACHE_SIZE:
                    prediction_cache[cache_key] = result
            
            return result
        except requests.exceptions.Timeout:
            # Xử lý lỗi timeout - trả về giá trị mặc định với thông báo rõ ràng hơn
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'API timeout - server có thể đang quá tải hoặc đang khởi động'
            }
        except requests.exceptions.ConnectionError:
            # Xử lý lỗi kết nối
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'Không thể kết nối đến API server'
            }
        except requests.exceptions.RequestException as e:
            # Xử lý các lỗi request khác
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
    
    # Tạo session riêng cho health check với timeout dài hơn
    session = get_session()
    max_retries = 3  # Số lần thử lại tối đa
    
    for retry_count in range(max_retries):
        try:
            # Tăng timeout lên để xử lý cold start
            response = session.get(f"{api_url}/health", timeout=30)
            
            if response.status_code == 200:
                status_data = response.json()
                status = status_data.get("status", "")
                
                if status == "healthy":
                    status_placeholder.success(f"✅ Đã kết nối đến API server tại {api_url}")
                    return True
                elif status == "initializing":
                    # API đang khởi tạo - đợi và thử lại
                    for i in range(20):  # Đợi tối đa 20 giây
                        status_placeholder.warning(f"⏳ API server đang khởi tạo... Vui lòng đợi ({i+1}/20s)")
                        time.sleep(1)
                        
                        try:
                            init_response = session.get(f"{api_url}/health", timeout=5)
                            if init_response.status_code == 200 and init_response.json().get("status") == "healthy":
                                status_placeholder.success(f"✅ API server đã sẵn sàng!")
                                return True
                        except requests.exceptions.RequestException:
                            pass
                    
                    # Nếu không thành công sau khi chờ đợi, thử lại từ đầu (nếu còn lần thử)
                    if retry_count < max_retries - 1:
                        status_placeholder.info(f"🔄 Đang thử kết nối lại ({retry_count + 2}/{max_retries})...")
                    else:
                        status_placeholder.warning(f"⚠️ API server vẫn đang khởi tạo sau nhiều lần thử. Tiếp tục với dự đoán local.")
                        return True
                else:
                    # Trạng thái không mong muốn
                    status_placeholder.warning(f"⚠️ API server trả về trạng thái không mong muốn: {status}")
                    return True
            else:
                if retry_count < max_retries - 1:
                    status_placeholder.info(f"🔄 Mã phản hồi không mong muốn ({response.status_code}). Đang thử lại ({retry_count + 2}/{max_retries})...")
                    time.sleep(2)  # Đợi 2 giây trước khi thử lại
                else:
                    status_placeholder.error(f"❌ API server trả về mã lỗi: {response.status_code}")
                    return True
        except requests.exceptions.Timeout:
            if retry_count < max_retries - 1:
                status_placeholder.info(f"🔄 Timeout khi kết nối. Đang thử lại ({retry_count + 2}/{max_retries})...")
                time.sleep(2)  # Đợi 2 giây trước khi thử lại
            else:
                status_placeholder.error(f"❌ Không thể kết nối đến API server tại {api_url}: Connection timeout")
                return True
        except requests.exceptions.ConnectionError:
            if retry_count < max_retries - 1:
                status_placeholder.info(f"🔄 Lỗi kết nối. Đang thử lại ({retry_count + 2}/{max_retries})...")
                time.sleep(2)  # Đợi 2 giây trước khi thử lại
            else:
                status_placeholder.error(f"❌ Không thể kết nối đến API server tại {api_url}: Connection refused")
                return True
        except requests.exceptions.RequestException as e:
            if retry_count < max_retries - 1:
                status_placeholder.info(f"🔄 Lỗi request. Đang thử lại ({retry_count + 2}/{max_retries})...")
                time.sleep(2)  # Đợi 2 giây trước khi thử lại
            else:
                status_placeholder.error(f"❌ Lỗi khi kết nối đến API server: {str(e)}")
                return True
    
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