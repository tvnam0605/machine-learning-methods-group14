import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import math
from ultralytics import YOLO
import os

model = YOLO(r'D:\machinelearning_methods\project\drive-drowsiness\models\last_train_100_epochs.pt')
classNames = ['EyeClosed', 'Eyeopen', 'Yawning']

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Drive Drowsiness Detection")
        self.root.geometry("900x700")
        self.root.configure(bg="#f4f4f4")

        self.title_label = tk.Label(root, text="Drive Drowsiness Detection", font=("Arial", 20, "bold"), bg="#f4f4f4", fg="#333")
        self.title_label.pack(pady=10)

        self.main_frame = tk.Frame(root, bg="#ffffff", relief="solid", bd=1)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.display_label = tk.Label(self.main_frame, bg="#eaeaea", text="Video/Ảnh", font=("Arial", 12), relief="groove")
        self.display_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(root, bg="#f4f4f4")
        self.button_frame.pack(pady=10)

        self.webcam_button = tk.Button(self.button_frame, text="Webcam", command=self.start_webcam, font=("Arial", 12), bg="#007BFF", fg="#ffffff", width=15)
        self.webcam_button.grid(row=0, column=0, padx=10, pady=5)

        self.stop_button = tk.Button(self.button_frame, text="Dừng Webcam", command=self.stop_webcam, font=("Arial", 12), bg="#FFC107", fg="#ffffff", width=15)
        self.stop_button.grid(row=0, column=1, padx=10, pady=5)

        self.upload_button = tk.Button(self.button_frame, text="Tải anh lên", command=self.upload_file, font=("Arial", 12), bg="#28A745", fg="#ffffff", width=15)
        self.upload_button.grid(row=0, column=2, padx=10, pady=5)

        self.exit_button = tk.Button(self.button_frame, text="Thoát", command=self.exit_app, font=("Arial", 12), bg="#DC3545", fg="#ffffff", width=15)
        self.exit_button.grid(row=0, column=3, padx=10, pady=5)

        self.result_frame = tk.Frame(root, bg="#f4f4f4")
        self.result_frame.pack(pady=10)
        self.result_label = tk.Label(self.result_frame, text="Kết quả nhận diện:", font=("Arial", 14, "bold"), bg="#f4f4f4", fg="#333")
        self.result_label.pack(anchor="w", padx=10)

        self.result_text = tk.Text(self.result_frame, height=10, width=80, font=("Arial", 12), bg="#ffffff", fg="#333", relief="solid", bd=1)
        self.result_text.pack(padx=10, pady=5)

        self.cap = None
        self.stop_webcam_flag = False

    def start_webcam(self):
        self.stop_webcam_flag = False
        self.cap = cv2.VideoCapture(0)

        def update_frame():
            if self.stop_webcam_flag:
                return
            success, img = self.cap.read()
            if success:
                img = self.detect_objects(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)
                self.display_label.imgtk = img_tk
                self.display_label.configure(image=img_tk)
            self.root.after(10, update_frame)

        update_frame()

    def stop_webcam(self):
        self.stop_webcam_flag = True
        if self.cap:
            self.cap.release()
            self.cap = None

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("Video Files", "*.mp4")])
        if file_path:
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.process_image(file_path)
            else:
                messagebox.showinfo("Unsupported File", "Currently, only image files are supported.")

    def process_image(self, file_path):
        img = cv2.imread(file_path)
        img = self.detect_objects(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        self.display_label.imgtk = img_tk
        self.display_label.configure(image=img_tk)

    def detect_objects(self, img):
        results = model(img, stream=True)
        self.result_text.delete(1.0, tk.END)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                if 0 <= cls < len(classNames):
                    class_name = classNames[cls]
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    label = f'{class_name} {confidence:.2f}'
                    self.result_text.insert(tk.END, f'{label}\n')
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

    def exit_app(self):
        self.stop_webcam()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()
