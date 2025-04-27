import streamlit as st
from utils.visualization import (
    plot_feature_importance,
    plot_emission_comparison,
    create_gauge_chart,
    style_metric_cards
)
import pandas as pd
import time
import numpy as np
from utils.benchmark_utils import BenchmarkUtils
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class MainView:
    """
    MainView là lớp chính quản lý giao diện người dùng của ứng dụng Streamlit
    Lớp này chịu trách nhiệm hiển thị các trang và tương tác với người dùng
    Kết nối với controller để thực hiện các dự đoán và phân tích dữ liệu
    """
    def __init__(self, controller):
        """
        Khởi tạo đối tượng MainView
        
        Parameters:
            controller: EmissionController - Đối tượng controller để xử lý logic nghiệp vụ và dự đoán
        """
        self.controller = controller
        self.benchmark_utils = BenchmarkUtils()

    def show(self):
        """
        Hiển thị giao diện chính của ứng dụng với thanh điều hướng bên và các trang tương ứng
        Người dùng có thể chuyển đổi giữa các trang: Dự đoán, Phân tích và Benchmark
        """
        # Thêm CSS tùy chỉnh để làm đẹp giao diện
        st.markdown(style_metric_cards(), unsafe_allow_html=True)
        
        # Thiết lập thanh điều hướng bên trái
        with st.sidebar:
            st.markdown("# 🚗 CO2 Emission Predictor")
            st.markdown("---")
            page = st.radio("Navigation", ["Prediction", "Analysis", "Benchmark"])

        # Hiển thị trang tương ứng theo lựa chọn người dùng
        if page == "Prediction":
            self._show_prediction_page()
        elif page == "Analysis":
            self._show_analysis_page()
        else:
            self._show_benchmark_page()

    def _show_prediction_page(self):
        """
        Hiển thị trang dự đoán phát thải CO2
        Cho phép người dùng nhập các đặc điểm của phương tiện và nhận dự đoán phát thải CO2
        Hiển thị kết quả dưới dạng số và biểu đồ trực quan
        """
        st.title("🌍 Predict Vehicle CO2 Emissions")
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px'>
            <h4 style='margin: 0; color: #0f4c81'>Enter your vehicle specifications to predict CO2 emissions</h4>
        </div>
        """, unsafe_allow_html=True)

        # Chia layout thành 2 cột để nhập thông tin
        col1, col2 = st.columns(2)

        # Cột bên trái cho các thông số đầu tiên
        with col1:
            engine_size = st.number_input("🔧 Engine Size (L)", 
                                        min_value=0.1, 
                                        max_value=10.0, 
                                        value=2.0,
                                        step=0.1)
            
            cylinders = st.number_input("⚙️ Number of Cylinders",
                                      min_value=2,
                                      max_value=16,
                                      value=4,
                                      step=1)
            
            fuel_consumption = st.number_input("⛽ Fuel Consumption (L/100 km)",
                                             min_value=1.0,
                                             max_value=30.0,
                                             value=8.0,
                                             step=0.1)

        # Cột bên phải cho các thông số còn lại
        with col2:
            horsepower = st.number_input("🏎️ Horsepower",
                                       min_value=50,
                                       max_value=1000,
                                       value=200,
                                       step=10)
            
            weight = st.number_input("⚖️ Vehicle Weight (kg)",
                                   min_value=500,
                                   max_value=5000,
                                   value=1500,
                                   step=100)
            
            year = st.number_input("📅 Vehicle Year",
                                 min_value=2015,
                                 max_value=2024,
                                 value=2023,
                                 step=1)

        # Nút dự đoán để kích hoạt quá trình dự đoán
        if st.button("🔍 Predict Emissions", type="primary"):
            # Tạo dictionary thông số xe để truyền vào controller
            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }

            try:
                # Thực hiện dự đoán và lấy các thông tin liên quan
                prediction = self.controller.predict_emission(features)
                avg_emission = self.controller.get_average_emission()
                rating = self.controller.get_emission_rating(prediction)
                tips = self.controller.get_eco_tips(prediction)

                # Hiển thị kết quả
                st.markdown("### 📊 Results")
                col1, col2, col3 = st.columns(3)
                
                # Cột 1: Kết quả dự đoán CO2
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>🎯 Predicted CO2 Emission</h3>
                            <div class="metric-value">{prediction:.1f} g/km</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Cột 2: Xếp hạng phát thải
                with col2:
                    rating_colors = {
                        'A': '🟢', 'B': '🟡', 'C': '🟠',
                        'D': '🔴', 'E': '🟣', 'F': '⚫'
                    }
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>📈 Emission Rating</h3>
                            <div class="metric-value">{rating_colors.get(rating, '⚪')} {rating}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Cột 3: So sánh với mức trung bình
                with col3:
                    comparison = ((prediction - avg_emission) / avg_emission * 100)
                    icon = "🔽" if comparison < 0 else "🔼"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>📊 Compared to Average</h3>
                            <div class="metric-value">
                                {icon} {'+' if comparison > 0 else ''}{comparison:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Hiển thị biểu đồ trực quan
                st.markdown("### 📈 Visualization")
                col1, col2 = st.columns(2)
                
                # Biểu đồ so sánh phát thải
                with col1:
                    st.pyplot(plot_emission_comparison(prediction, avg_emission))
                
                # Biểu đồ đồng hồ đo
                with col2:
                    st.pyplot(create_gauge_chart(prediction, 0, 300, "Emission Meter"))

                # Hiển thị mẹo thân thiện môi trường
                st.markdown("### 🌱 Eco-friendly Tips")
                for tip in tips:
                    st.markdown(f"- {tip}")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    def _show_analysis_page(self):
        """
        Hiển thị trang phân tích các tính năng quan trọng ảnh hưởng đến phát thải CO2
        Hiển thị biểu đồ độ quan trọng của các đặc trưng trong mô hình dự đoán
        """
        st.title("📊 CO2 Emission Analysis")
        
        # Phân tích độ quan trọng của từng đặc trưng
        st.subheader("🎯 Feature Importance Analysis")
        try:
            # Lấy thông tin độ quan trọng của các đặc trưng từ controller
            importance_dict = self.controller.get_feature_importance()
            st.pyplot(plot_feature_importance(importance_dict))
            
            # Thêm giải thích về biểu đồ độ quan trọng
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 20px'>
                <h4 style='margin: 0; color: #0f4c81'>Understanding Feature Importance</h4>
                <p style='margin-top: 10px'>
                    This chart shows how much each vehicle characteristic influences CO2 emissions. 
                    Longer bars indicate stronger influence on the prediction.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")

        # Phần này có thể mở rộng để thêm các phân tích khác

    def _show_benchmark_page(self):
        """
        Hiển thị trang benchmark để kiểm tra hiệu suất của API
        Cho phép người dùng thực hiện 1000 request đến API để đánh giá thời gian đáp ứng
        Hỗ trợ hai chế độ: tham số cố định hoặc tham số ngẫu nhiên
        """
        st.title("⏱️ Benchmark 1000 Requests")
        
        # Lấy URL API từ biến môi trường hoặc sử dụng giá trị mặc định
        API_URL = os.environ.get('API_URL', 'https://thuco2tiep.onrender.com')
        st.info(f"Using API endpoint: {API_URL}")
        
        # Kiểm tra trạng thái khả dụng của API
        try:
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code == 200:
                st.success("API is healthy and ready!")
            else:
                st.warning(f"API health check failed: {health_response.json().get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Could not connect to API: {str(e)}")
            return

        # Lựa chọn chế độ kiểm tra: Tham số cố định hoặc tham số ngẫu nhiên
        test_mode = st.radio(
            "Chế độ kiểm tra",
            ["Tham số cố định", "Tham số ngẫu nhiên"]
        )

        # Hiển thị form nhập thông số cho chế độ tham số cố định
        if test_mode == "Tham số cố định":
            st.subheader("Nhập tham số kiểm tra:")
            col1, col2 = st.columns(2)
            
            # Cột bên trái cho các thông số đầu tiên
            with col1:
                engine_size = st.number_input("Engine Size (L)", 
                    min_value=0.1, max_value=10.0, value=2.0, step=0.1)
                cylinders = st.number_input("Cylinders", 
                    min_value=2, max_value=16, value=4, step=1)
                fuel_consumption = st.number_input("Fuel Consumption (L/100km)", 
                    min_value=1.0, max_value=30.0, value=8.0, step=0.1)
            
            # Cột bên phải cho các thông số còn lại
            with col2:
                horsepower = st.number_input("Horsepower", 
                    min_value=50, max_value=1000, value=200, step=10)
                weight = st.number_input("Weight (kg)", 
                    min_value=500, max_value=5000, value=1500, step=100)
                year = st.number_input("Year", 
                    min_value=2015, max_value=2024, value=2023, step=1)

            # Tạo dictionary các thông số xe
            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }
        else:
            # Nếu chọn chế độ tham số ngẫu nhiên, tạo mẫu tham số ngẫu nhiên
            features = self.generate_random_features()
            st.info("Mỗi request sẽ sử dụng một bộ tham số ngẫu nhiên khác nhau")
            st.write("Ví dụ tham số ngẫu nhiên:", features)
            
        # Tùy chọn cho warm-up API (mặc định checked)
        do_warmup = st.checkbox("Khởi động API trước khi benchmark (khuyên dùng)", value=True)
        
        # Nút kích hoạt quá trình benchmark
        if st.button("Chạy Benchmark"):
            # Tạo container cho log và thanh tiến trình
            log_container = st.empty()
            progress_bar = st.progress(0)
            
            # Khởi động API trước khi benchmark nếu được chọn
            if do_warmup:
                log_container.info("Đang khởi động API server (warm-up)...")
                try:
                    # Gửi request health check với timeout dài hơn
                    requests.get(f"{API_URL}/health", timeout=30)
                    
                    # Gửi một request dự đoán đơn lẻ để khởi tạo mô hình
                    warm_up_response = requests.post(
                        f"{API_URL}/predict",
                        json=features,
                        timeout=60  # Chờ lâu hơn cho request đầu tiên
                    )
                    
                    if warm_up_response.status_code == 200:
                        result = warm_up_response.json()
                        log_container.success(f"API đã sẵn sàng! Kết quả warm-up: {result.get('prediction', 'N/A')} g/km")
                        # Đợi thêm 2 giây để đảm bảo API hoàn toàn sẵn sàng
                        time.sleep(2)
                    else:
                        log_container.warning("API trả về lỗi khi warm-up, tiếp tục benchmark với thận trọng.")
                except Exception as e:
                    log_container.warning(f"Không thể warm-up API: {str(e)}. Tiếp tục benchmark...")
            
            # Bắt đầu đo thời gian
            start_time = time.perf_counter()
            
            # Thiết lập thông số cho benchmark thực hiện 1000 request
            n_requests = 1000
            successful_requests = 0
            completed_requests = 0
            
            # Khởi tạo benchmark utils và bắt đầu phiên benchmark mới
            self.benchmark_utils.start_benchmark()
            
            # Danh sách lưu kết quả chi tiết
            benchmark_results = []
            
            # Biến lưu trữ thông tin về first request để debug
            first_request_info = None

            # Hàm thực hiện một request đến API
            def make_request(request_number):
                nonlocal first_request_info
                
                # Thiết lập timeout động: dài hơn cho request đầu tiên
                timeout = 30 if request_number == 0 else 10
                
                try:
                    # Tạo tham số: cố định hoặc ngẫu nhiên tùy chế độ đã chọn
                    request_features = (
                        self.generate_random_features() 
                        if test_mode == "Tham số ngẫu nhiên" 
                        else features
                    )
                    
                    # Gọi API với timeout và đo thời gian
                    req_start_time = time.perf_counter()
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=request_features,
                        timeout=timeout  # Timeout động
                    )
                    req_end_time = time.perf_counter()
                    total_time_ms = (req_end_time - req_start_time) * 1000  # ms
                    total_time_sec = (req_end_time - req_start_time)  # seconds
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Tính toán thời gian xử lý và mạng
                        processing_time_ms = result.get('process_time_ms', 0)
                        processing_time_sec = processing_time_ms / 1000  # Convert ms to seconds
                        network_time_sec = total_time_sec - processing_time_sec if total_time_sec > processing_time_sec else 0
                        
                        # Lưu thông tin chi tiết request
                        timing_data = {
                            'timestamp': pd.Timestamp.now(),
                            'request_number': request_number,
                            'total_time': total_time_sec,  # Seconds
                            'network_time': network_time_sec,  # Seconds
                            'processing_time': processing_time_sec,  # Seconds
                            'prediction': result.get('prediction', 0),
                            'status': result.get('status', 'success'),
                            'error': None
                        }
                        benchmark_results.append(timing_data)
                        
                        # Lưu thông tin request đầu tiên cho debug
                        if request_number == 0:
                            first_request_info = {
                                'features': request_features,
                                'prediction': result['prediction'],
                                'api_process_time': f"{processing_time_sec:.3f}s ({processing_time_ms}ms)",
                                'total_time': f"{total_time_sec:.3f}s",
                                'network_latency': f"{network_time_sec:.3f}s"
                            }
                        return True
                    else:
                        # Lưu thông tin về request thất bại
                        timing_data = {
                            'timestamp': pd.Timestamp.now(),
                            'request_number': request_number,
                            'total_time': total_time_sec,  # Seconds
                            'network_time': 0,
                            'processing_time': 0,
                            'prediction': 0,
                            'status': 'error',
                            'error': f"HTTP {response.status_code}"
                        }
                        benchmark_results.append(timing_data)
                        
                        # Lưu thông tin lỗi request đầu tiên
                        if request_number == 0:
                            first_request_info = {
                                'features': request_features,
                                'error': f"HTTP Error: {response.text}"
                            }
                        return False
                        
                except Exception as e:
                    # Lưu thông tin về request lỗi
                    timing_data = {
                        'timestamp': pd.Timestamp.now(),
                        'request_number': request_number,
                        'total_time': 0,
                        'network_time': 0,
                        'processing_time': 0,
                        'prediction': 0,
                        'status': 'error',
                        'error': str(e)
                    }
                    benchmark_results.append(timing_data)
                    
                    # Lưu thông tin lỗi request đầu tiên
                    if request_number == 0:
                        first_request_info = {
                            'features': request_features,
                            'error': f"Request Error: {str(e)}"
                        }
                    return False

            # Sử dụng ThreadPoolExecutor với chiến lược phân đợt
            with ThreadPoolExecutor(max_workers=50) as executor:
                # Chia thành các batch để tránh quá tải server
                batch_size = 100  # Mỗi đợt 100 request
                
                # Xử lý từng batch một
                for batch_start in range(0, n_requests, batch_size):
                    batch_end = min(batch_start + batch_size, n_requests)
                    batch_count = batch_end - batch_start
                    
                    # Gửi batch requests
                    future_to_request = {
                        executor.submit(make_request, i): i 
                        for i in range(batch_start, batch_end)
                    }
                    
                    # Xử lý kết quả khi các request trong batch hoàn thành
                    for future in as_completed(future_to_request):
                        completed_requests += 1
                        if future.result():
                            successful_requests += 1
                        
                        # Cập nhật thanh tiến trình
                        progress = completed_requests / n_requests
                        progress_bar.progress(progress)
                        
                        # Cập nhật log mỗi 50 request
                        if completed_requests % 50 == 0 or completed_requests == n_requests:
                            current_time = time.perf_counter() - start_time
                            log_container.info(
                                f"Đã xử lý {completed_requests}/{n_requests} requests... "
                                f"({current_time:.1f}s), {successful_requests} thành công"
                            )
                    
                    # Đợi một chút giữa các batch để tránh quá tải server
                    if batch_end < n_requests:
                        time.sleep(0.5)
            
            # Hiển thị thông tin request đầu tiên nếu có
            if first_request_info:
                if 'error' in first_request_info:
                    st.error(f"Debug - First request error: {first_request_info['error']}")
                else:
                    st.write("Debug - First request:", first_request_info)
            
            # Kết thúc đo thời gian
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Kết thúc phiên benchmark và lưu kết quả
            self.benchmark_utils.end_benchmark()
            
            # Hiển thị kết quả benchmark
            st.success("Benchmark hoàn thành!")
            st.markdown(f"""
            ### Kết quả:
            - Chế độ kiểm tra: {test_mode}
            - Tổng thời gian: {total_time:.2f} giây
            - Số request thành công: {successful_requests}/{n_requests}
            - Tốc độ trung bình: {n_requests/total_time:.1f} requests/giây
            """)
            
            # Hiển thị bảng kết quả chi tiết từ benchmark_utils
            st.markdown("### Bảng chi tiết kết quả benchmark:")
            
            # Lưu kết quả vào đối tượng benchmark_utils
            self.benchmark_utils.results = benchmark_results
            
            # Sử dụng hàm get_results_df để lấy DataFrame kết quả
            results_df = self.benchmark_utils.get_results_df()
            
            # Lấy mẫu để hiển thị (tối đa 100 dòng)
            if len(results_df) > 100:
                results_df = results_df.sample(n=100).sort_values('request_number')
                st.info(f"Hiển thị 100 mẫu ngẫu nhiên từ tổng số {len(benchmark_results)} requests")
            
            # Thêm thông tin về đơn vị đo
            st.markdown("**Lưu ý**: Thời gian trong bảng được đo bằng đơn vị **giây (s)**")
            
            # Hiển thị bảng với định dạng
            st.dataframe(
                results_df, 
                use_container_width=True,
                hide_index=True
            )

    def generate_random_features(self):
        """
        Tạo bộ tham số ngẫu nhiên cho phương tiện
        
        Returns:
            dict: Dictionary chứa các tham số ngẫu nhiên của phương tiện
        """
        return {
            'Engine Size(L)': np.random.uniform(1.0, 8.0),
            'Cylinders': np.random.randint(3, 12),
            'Fuel Consumption Comb (L/100 km)': np.random.uniform(4.0, 20.0),
            'Horsepower': np.random.uniform(100, 800),
            'Weight (kg)': np.random.uniform(1000, 4000),
            'Year': np.random.randint(2015, 2024)
        } 