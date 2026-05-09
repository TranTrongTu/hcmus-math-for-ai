# 🏡 Lab 01: Dự đoán giá nhà với Hồi quy Tuyến tính (House Price Prediction)

## 📝 Giới thiệu
Dự án này là Đồ án 1 thuộc môn **Phương pháp Toán cho Trí tuệ Nhân tạo**. Mục tiêu của dự án là xây dựng mô hình Hồi quy Tuyến tính (Linear Regression) để dự đoán giá nhà (SalePrice) dựa trên các đặc trưng của ngôi nhà. Điểm nhấn của dự án là việc **tự cài đặt (from scratch) các thuật toán tối ưu toán học** mà không phụ thuộc vào các thư viện Machine Learning bậc cao, nhằm hiểu sâu về cơ chế hoạt động cốt lõi của mô hình.

## ✨ Điểm nổi bật về Kỹ thuật
- **Cài đặt từ đầu (From Scratch):** Triển khai thuật toán **Gradient Descent** hoàn toàn bằng `NumPy` để tính toán ma trận và cập nhật trọng số.
- **Phân tích Dữ liệu Khám phá (EDA):** Sử dụng `pandas`, `matplotlib`, và `seaborn` để trực quan hóa, phân tích phân phối của biến mục tiêu và ma trận tương quan giữa các đặc trưng.
- **Xử lý Ngoại lệ (Outlier Handling):** Đi sâu vào phân tích lỗi (Error Analysis) thông qua việc trực quan hóa sự chênh lệch giữa giá trị thực tế và dự đoán (Actual vs Predicted). So sánh chi tiết các biến thể mô hình (Model 2a, 2b, 2c) để đánh giá mức độ ảnh hưởng của các điểm dữ liệu bất thường.
- **Đánh giá Mô hình:** Đánh giá độ tin cậy của thuật toán thông qua các metric tự xây dựng và phân tích trực quan phần dư (residuals).

## 📂 Cấu trúc Thư mục
- `dataset/`: Chứa dữ liệu huấn luyện (`train.csv`), dữ liệu kiểm thử (`test.csv`) và file giải thích các biến (`data_description.txt`).
- `24122024.ipynb`: File Jupyter Notebook chính chứa toàn bộ mã nguồn: từ bước làm sạch dữ liệu, EDA, cài đặt thuật toán đến đánh giá mô hình.
- `report.pdf`: Báo cáo chi tiết về cơ sở toán học, đạo hàm các hàm mất mát và phân tích kết quả thực nghiệm.

## 🚀 Hướng dẫn Cài đặt và Sử dụng

1. **Clone repository:**
   ```bash
   git clone [https://github.com/TranTrongTu/hcmus-math-for-ai.git](https://github.com/TranTrongTu/hcmus-math-for-ai.git)
   cd hcmus-math-for-ai/lab01
   ```

2. **Cài đặt môi trường:**
   Đảm bảo bạn đã cài đặt Python. Sau đó, cài đặt các thư viện cần thiết thông qua file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy Jupyter Notebook:**
   Mở notebook để xem các bước huấn luyện và biểu đồ phân tích:
   ```bash
   jupyter notebook 24122024.ipynb
   ```

## 📈 Tóm tắt Kết quả
Mô hình đã thể hiện sự hội tụ ổn định thông qua thuật toán Gradient Descent tự cài đặt. Quá trình đối chiếu (Actual vs Predicted) kết hợp với việc khoanh vùng các điểm ngoại lệ (Outliers) cho thấy việc tinh chỉnh hàm mất mát toán học và tối ưu hóa đặc trưng đóng vai trò quyết định trong việc cải thiện độ chính xác dự đoán của mô hình.