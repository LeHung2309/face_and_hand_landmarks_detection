# Phát hiện Landmark và Dự đoán Tuổi (Real-time)

Dự án này sử dụng thư viện **MediaPipe** và **OpenCV** để thực hiện nhận dạng các điểm mốc (landmarks) trên khuôn mặt, bàn tay, tư thế cơ thể và **dự đoán tuổi** trong thời gian thực thông qua webcam. Hệ thống được tối ưu hóa đa luồng (multi-threading) để đảm bảo hiệu năng cao.

## Tính năng

- **Nhận dạng điểm mốc toàn diện (Holistic):**
  - Khuôn mặt (Face Mesh)
  - Bàn tay (Hand Landmarks)
  - Tư thế (Pose Landmarks)
- **Dự đoán tuổi:** Ước tính độ tuổi của khuôn mặt được phát hiện bằng mô hình Deep Learning (Caffe).
- **Hiệu năng cao:** Xử lý AI trên các luồng riêng biệt (background threads) để không làm gián đoạn luồng video.
- **Hiển thị thông tin:** FPS (tốc độ khung hình) và tuổi được hiển thị trực tiếp.

## Yêu cầu hệ thống

- Python 3.8 trở lên.
- Webcam.

## Cài đặt

1. **Khởi tạo môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv venv
   
   # Trên Windows
   .\venv\Scripts\activate
   
   # Trên Linux/macOS
   source venv/bin/activate
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Kiểm tra Model:**
   Đảm bảo thư mục `models/` chứa các file sau:
   - `age_deploy.prototxt`
   - `age_net.caffemodel`

## Cách sử dụng

Chạy file `main.py` để bắt đầu ứng dụng:

```bash
python main.py
```

- Nhấn **'q'** để thoát ứng dụng.

## Cấu trúc mã nguồn

- **`main.py`**: Điểm khởi chạy chính. Quản lý vòng lặp hiển thị và điều phối các luồng xử lý.
- **`config.py`**: Chứa các lớp cấu hình (Camera, Model, MediaPipe).
- **`camera_utils.py`**: Xử lý việc đọc camera trong luồng riêng để tối ưu FPS.
- **`detectors.py`**: Wrapper cho MediaPipe Holistic, chạy trên luồng riêng.
- **`age_predictor.py`**: Xử lý dự đoán tuổi sử dụng OpenCV DNN, chạy trên luồng riêng.
- **`models/`**: Chứa các mô hình đã huấn luyện.

## Thư viện chính
- [MediaPipe](https://mediapipe.dev/): Nhận diện điểm mốc cơ thể.
- [OpenCV](https://opencv.org/): Xử lý hình ảnh và chạy mô hình DNN.
- [NumPy](https://numpy.org/): Tính toán số học.