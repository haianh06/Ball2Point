# ⚽ Ball2Point: AI-Powered Football Tactical Analysis

**Ball2Point** là một hệ thống phân tích chiến thuật bóng đá toàn diện dựa trên Trí tuệ Nhân tạo (Computer Vision & Machine Learning). Hệ thống tự động bóc tách luồng video camera đơn (Single-camera Broadcast) thành các ma trận dữ liệu không gian, từ đó cung cấp các góc nhìn chuyên sâu về chiến thuật, thể lực và quyền kiểm soát sân.

---

## 🚀 1. Tính năng cốt lõi (Key Features)

Dự án được thiết kế theo mô hình Micro-Modules độc lập, tích hợp thông qua một Orchestrator duy nhất:

* **Module 1 - Detection & Tracking:** Nhận diện Cầu thủ/Trọng tài/Bóng. Tự động chia đội (Team Assignment) mà không cần cấu hình màu áo trước. Khử nhiễu và nội suy quỹ đạo bóng.
* **Module 2 - Tactical Minimap (2D Mapping):** Chiếu phối cảnh (Perspective Transform) từ pixel camera xuống mặt phẳng 2D chuẩn hệ mét của FIFA.
* **Module 3 - Speed & Distance Profiling:** Đo lường vận tốc tức thời và tổng quãng đường di chuyển của từng cầu thủ, tự động lọc nhiễu dao động (Jitter Filtering).
* **Module 4 - Physical Heatmap:** Kết xuất bản đồ nhiệt thể lực cho toàn bộ cầu thủ sau trận đấu.
* **Module 5 - Pitch Control (Voronoi):** Phân tích không gian kiểm soát tĩnh của hai đội dựa trên mạng lưới đa giác Voronoi.
* **Dashboard UI (Streamlit):** Giao diện tương tác trực quan, hỗ trợ chunking upload video dung lượng lớn, Live Preview và render H.264.

---

## 🧠 2. Kiến trúc thuật toán (Algorithms & Under the Hood)

* **Object Detection:** Sử dụng `YOLO` (Ultralytics) để detect Cầu thủ, Trọng tài và Bóng.
* **Multi-Object Tracking:** Sử dụng thuật toán `ByteTrack` (thư viện Supervision) để duy trì ID cầu thủ qua các frame.
* **Team Clustering (Zero-shot):** Trích xuất đặc trưng hình ảnh (Feature Extraction) bằng mạng Vision Transformer siêu nhẹ `SigLIP` của Google, giảm chiều dữ liệu bằng `UMAP`, và phân cụm không giám sát bằng `K-Means`.
* **Homography (Spatial Math):** Nhận diện 29 điểm chuẩn (Keypoints) trên sân bóng bằng mô hình YOLO Pose. Sử dụng thuật toán `RANSAC` (`cv2.findHomography`) để tìm ma trận chiếu. Tích hợp cơ chế Fallback-State đóng băng khung hình khi mất điểm neo.
* **Speed Profiling:** Tính toán trên một cửa sổ trượt (Sliding Window / Batch processing) 5-frames để triệt tiêu nhiễu pixel, giới hạn ngưỡng vật lý (Usain Bolt threshold) ở mức 40km/h.
* **Heatmap Generation:** Tích lũy ma trận 2D Float32, làm mượt bằng `cv2.GaussianBlur` và phủ dải màu nhiệt (Colormap) bằng kỹ thuật Dynamic Alpha Masking.
* **Voronoi Tesselation:** Sử dụng lõi C++ `cv2.Subdiv2D` để chia lưới tam giác Delaunay và trích xuất đa giác Voronoi siêu tốc.

---

## 📂 3. Tổ chức thư mục (Folder Structure)

```text
Ball2Point/
├── Models/
│   └── weights/
│       ├── best_detection.pt         # Model nhận diện cầu thủ, bóng
│       └── best_keypoints.pt         # Model nhận diện 29 điểm neo sân bóng
├── inputs/                           # Thư mục chứa video gốc
├── outputs/
│   ├── player_heatmaps/              # Ảnh Heatmap thể lực xuất ra
│   ├── final_dashboard.mp4           # Video kết quả H.264 cho Web
│   └── player_statistics.json        # Dữ liệu quãng đường, vận tốc
├── module_1/                         # Core: Detection, Tracking, Clustering
├── module_2/                         # Core: Homography & 2D Minimap
├── module_3/                         # Core: Speed & Distance Math
├── module_4/                         # Core: Heatmap Accumulation
├── module_5/                         # Core: Voronoi Pitch Control
├── .streamlit/
│   └── config.toml                   # Cấu hình UI Theme & Mở khóa Upload Limit
├── .env                              # Biến môi trường
├── app.py                            # Streamlit Web Dashboard (Entrypoint)
└── requirements.txt                  # Danh sách thư viện
```

---

## 🛠️ 4. Chuẩn bị & Cài đặt (Setup & Installation)

### Yêu cầu phần cứng / Hệ thống:
* **Hệ điều hành:** Windows / Linux / macOS.
* **GPU:** Khuyến nghị card đồ họa NVIDIA (có CUDA).

### Các bước cài đặt:

**Bước 0: Cài đặt FFmpeg (Bắt buộc)**
Hệ thống yêu cầu FFmpeg ở cấp độ OS (Operating System) để re-encode video sang chuẩn H.264 (giúp trình duyệt Web có thể phát được video).

* **1. Dành cho Windows:** Mở **PowerShell** (Run as Administrator) và chạy lệnh sau:
  ```bash
  winget install ffmpeg
  ```
  *(Lưu ý: Sau khi cài xong, hãy khởi động lại Terminal/VS Code để máy tính nhận diện biến môi trường PATH).* Hoặc cài thủ công qua [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) và thêm vào System PATH.

* **2. Dành cho macOS:**
  Mở Terminal và chạy:
  ```bash
  brew install ffmpeg
  ```

* **3. Dành cho Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```

*Kiểm tra cài đặt: Mở terminal mới và gõ lệnh `ffmpeg -version`. Nếu in ra thông số phiên bản thì hệ thống đã sẵn sàng.*

**Bước 1: Clone dự án và tạo môi trường ảo (Virtual Environment)**
```bash
git clone https://github.com/your-username/Ball2Point.git
cd Ball2Point
python -m venv venv

# Active môi trường ảo
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

**Bước 2: Cài đặt PyTorch chuẩn GPU (CUDA 12.1)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Bước 3: Cài đặt các thư viện lõi**
Chạy `pip install -r requirements.txt`:

**Bước 4: Thiết lập Models**
* Đặt các file pre-trained weights (`best_detection.pt`, `best_keypoints.pt`) vào thư mục `Models/weights/`.
* (Tùy chọn) Khởi tạo file `.env` nếu cần tinh chỉnh đường dẫn.

---

## 💻 5. Hướng dẫn sử dụng (Usage)

Khởi động hệ thống Dashboard bằng lệnh:
```bash
streamlit run app.py
```
1. Truy cập vào `http://localhost:8501` trên trình duyệt.
2. Tại thanh Sidebar, upload video trận đấu (hỗ trợ MP4, AVI, MOV không giới hạn dung lượng).
3. Bật/Tắt các Module phân tích mong muốn (Minimap, Tốc độ, Heatmap, Voronoi).
4. Nhấn **XỬ LÝ DỮ LIỆU** và theo dõi Live Preview.
5. Xem kết quả chi tiết tại 3 Tabs: Video Chiến thuật, Bảng Thông số Thể lực, Thư viện Bản đồ Nhiệt.

---

## 🎯 6. Ứng dụng thực tiễn (Applications)

* **Coaching Staff (Ban huấn luyện):** Đánh giá hiệu suất di chuyển, phát hiện các khoảng trống chiến thuật bị bỏ lỡ (thông qua Voronoi), và kiểm soát cường độ vận động của cầu thủ để tránh chấn thương (Heatmap & Sprint log).
* **Broadcasting (Truyền hình):** Cung cấp các frame hình phân tích nâng cao, tăng tính hấp dẫn cho các bình luận viên trong giờ nghỉ giữa hiệp hoặc sau trận đấu.
* **Scouting (Tuyển trạch):** Số hóa các thuộc tính vật lý của cầu thủ (Top speed, Work rate) dựa trên video thi đấu thực tế mà không cần thiết bị GPS đeo trên người (Wearable trackers).

---
