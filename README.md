# Nhận diện hình ảnh: Cảnh báo buồn ngủ khi lái xe & Dự đoán bệnh ở cây trồng

## Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Hướng Dẫn Cài Đặt](#hướng-dẫn-cài-đặt)
4. [Hướng Dẫn Chạy](#hướng-dẫn-chạy)
5. [Đóng Góp](#đóng-góp)
6. [Liên Hệ](#liên-hệ)

## Giới thiệu
Dự án này bao gồm hai ứng dụng cụ thể sử dụng công nghệ nhận diện hình ảnh:

1. **Cảnh báo buồn ngủ khi lái xe**: Phát hiện trạng thái buồn ngủ của tài xế dựa trên hình ảnh khuôn mặt (mắt, miệng) được thu thập từ camera và đưa ra cảnh báo kịp thời.

2. **Dự đoán bệnh ở cây trồng**: Nhận diện và chẩn đoán các loại bệnh phổ biến ở cây trồng thông qua hình ảnh lá cây.

---

## Yêu cầu hệ thống

- **Python**: >= 3.8
- **Thư viện Python**:
  - OpenCV
  - TensorFlow
  - Keras
  - NumPy
  - Matplotlib
  - Scikit-learn
- **Phần cứng**:
  - Camera (cho ứng dụng cảnh báo buồn ngủ)
  - Ảnh mẫu lá cây (cho ứng dụng dự đoán bệnh cây trồng)

---

## Hướng dẫn cài đặt

1. Clone repository:
   ```bash
   git clone https://github.com/tvnam0605/machine-learning-methods-group14.git
   cd machine-learning-methods-group14
2. **Cài đặt các thư viện liên quan**:
   - **Tạo môi trường ảo**:
     ```bash
     python -m venv env
     source env/bin/activate      # Trên Linux/MacOS
     .\env\Scripts\activate       # Trên Windows
     ```
   - **Cài đặt các thư viện từ tệp `requirements.txt`**:
     ```bash
     pip install -r requirements.txt
     ```
3. Chọn ứng dụng để chạy
   - **Để chạy ứng dụng cảnh báo buồn ngủ**:
     ```bash
     cd drive-drowsiness
     ```
   - **Để chạy ứng dụng dự đoán bệnh ở cây trồng**:
     ```bash
     cd plant-detect
     ```
## Hướng dẫn chạy
1. Cảnh báo buồn ngủ khi lái xe
    ```bash
    cd drive-drowsiness
    python app-tkinter.py  # chạy app tkinter
    python app.py          # chạy webapp với flask
2. Dự đoán bệnh ở cây trồng
    ```bash
    cd plant-detect/app
    python main.py  # chạy webapp
  -**Đối với Dự đoán bệnh ở cây trồng**
  [Tải model tại đây]([https://link-google-drive-cua-ban](https://drive.google.com/drive/folders/18UxVZ4qUlmlw8Qv4RzGmNocFWONkJT7j?usp=sharing))
## Đóng góp
1. Trần Văn Nam - 2115239 - Ứng dụng cảnh báo buồn ngủ khi lái xe
2. Trương Tấn Diệm - 2111817 - Dự đoán bệnh ở cây trồng
## Liên hệ
  -**trannamvan0605@gmail.com**
