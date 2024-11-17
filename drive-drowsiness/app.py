from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import math
from ultralytics import YOLO
import os

app = Flask(__name__)

# Tải mô hình YOLO
model = YOLO(r'D:\machinelearning_methods\project\drive-drowsiness\models\last_train_100_epochs.pt')
classNames = ['EyeClosed', 'Eyeopen', 'Yawning']

# Tạo thư mục lưu ảnh và video tải lên
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Video feed từ webcam
def gen_frames():
    cap = cv2.VideoCapture(0)  # Mở webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Giảm độ phân giải
    while True:
        success, img = cap.read()
        if not success:
            break

        # Nhận kết quả từ mô hình YOLO
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Kiểm tra lớp và thay đổi màu sắc của khung
                cls = int(box.cls[0])
                if 0 <= cls < len(classNames):
                    class_name = classNames[cls]
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    label = f'{class_name} {confidence:.2f}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, [255, 228, 181], -1)
                    cv2.putText(img, label, (x1, y1-2), 0, 1, [85, 107, 47], thickness=2, lineType=cv2.FONT_HERSHEY_SIMPLEX)

                    # Chọn màu sắc của khung tùy theo trạng thái
                    if class_name == 'EyeClosed':  # Mắt nhắm
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Màu đỏ
                    elif class_name == 'Yawning':  # Ngáp
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)  # Màu cam
                    else:  # Mắt mở
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Màu xanh lá

        # Chuyển đổi ảnh thành định dạng mà Flask có thể hiển thị
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            continue  # Nếu không mã hóa được, bỏ qua frame này

        frame = buffer.tobytes()

        # Trả về frame trong định dạng video MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route chính
@app.route('/')
def index():
    return render_template('index.html')

# Route video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route tải video hoặc ảnh
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            # Xử lý ảnh/video tải lên và trả về kết quả
            result_text, result_image_path = process_uploaded_file(filename)
            return render_template('result.html', result=result_text, image_path=os.path.basename(result_image_path))

    return render_template('upload.html')

# Hàm xử lý file tải lên (video hoặc ảnh)
def process_uploaded_file(filepath):
    # Đọc ảnh
    img = cv2.imread(filepath)

    # Nhận kết quả từ mô hình YOLO
    results = model(img)
    
    result_text = []  # Danh sách lưu các kết quả nhận diện

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Kiểm tra lớp và thay đổi màu sắc của khung
            cls = int(box.cls[0])
            if 0 <= cls < len(classNames):
                class_name = classNames[cls]
                confidence = math.ceil((box.conf[0] * 100)) / 100
                label = f'{class_name} {confidence:.2f}'

                # Thêm kết quả nhận diện vào danh sách
                result_text.append(f'Class: {class_name}, Confidence: {confidence}')

                # Vẽ khung cho kết quả nhận diện
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Lưu ảnh kết quả
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + os.path.basename(filepath))
    cv2.imwrite(result_image_path, img)

    return result_text, result_image_path  # Trả về kết quả nhận diện và đường dẫn ảnh đã xử lý

# Route hiển thị kết quả nhận diện
@app.route('/result')
def result():
    return render_template('result.html', result=None)  # Hiển thị kết quả nhận diện

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
