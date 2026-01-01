# Phát hiện Landmark Khuôn mặt và Bàn tay (Real-time)

Dự án này sử dụng thư viện MediaPipe và OpenCV để thực hiện nhận dạng các điểm mốc (landmarks) trên khuôn mặt, bàn tay và tư thế cơ thể trong thời gian thực thông qua webcam.

## Tính năng
- Nhận dạng điểm mốc khuôn mặt (Face Mesh).
- Nhận dạng điểm mốc bàn tay trái và phải (Hand Landmarks).
- Nhận dạng tư thế cơ thể (Pose Landmarks).
- Hiển thị tốc độ khung hình (FPS) trực tiếp trên màn hình.
- Xử lý mượt mà với MediaPipe Holistic.

## Yêu cầu hệ thống
- Python 3.8 trở lên.
- Webcam tích hợp hoặc rời.

## Cài đặt

1. **Khởi tạo môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Linux/macOS
   # Hoặc
   .\venv\Scripts\activate  # Trên Windows
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

## Cách sử dụng

Chạy file `detection.py` để bắt đầu quá trình nhận dạng:

```bash
python detection.py
```

- Nhấn **'q'** để thoát ứng dụng.
- Cửa sổ camera sẽ hiển thị các điểm mốc được vẽ đè lên khuôn mặt, bàn tay và cơ thể của bạn.

## Cấu trúc mã nguồn
- `detection.py`: File chính chứa lớp `HolisticDetector` và logic xử lý luồng video.
- `requirements.txt`: Danh sách các thư viện phụ thuộc.

## Thư viện chính sử dụng
- [MediaPipe](https://mediapipe.dev/): Cung cấp các mô hình học máy cho thị giác máy tính.
- [OpenCV](https://opencv.org/): Xử lý hình ảnh và video.
- [NumPy](https://numpy.org/): Tính toán số học.
