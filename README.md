# Smart Attendance Management System Using Face Recognition
### Semester VII — Computer Vision Mini-Project

A complete, working college mini-project built for the **Semester VII Computer Vision** syllabus. The system combines modern deep-learning face recognition with classical digital image processing, persistent relational database storage, and a clean Streamlit interface.

---

## 🌟 Key Features

1. **Student Registration**:
   - Register students with Student ID, Name, and Class.
   - Capture via webcam or upload photos.
   - Automatic validation: Rejects images with 0 faces, multiple faces, small faces (<50×50), or low detection confidence.
   - Extracts 128-dimensional biometric embeddings and stores them securely in SQLite.
   - Directory viewer and deletion management with cascade cleanup.

2. **Face Recognition Attendance**:
   - Real-time face detection using **OpenCV YuNet** (anchor-free CNN with 5 facial keypoints).
   - Deep metric matching using **OpenCV SFace** with cosine similarity.
   - Visual annotations: Green boxes for recognized students, orange/red for unknown individuals.
   - Confirmation prompt before recording attendance.
   - **Duplicate Prevention**: Enforces a database composite constraint allowing exactly one attendance record per student per calendar date.
   - Strict rejection of unknown or low-confidence faces.

3. **Attendance Records & Reporting**:
   - Filter attendance logs by date, class, and student search.
   - Real-time KPI metrics: Enrolled, Present, Absent, Attendance Percentage.
   - Absenteeism inspector showing absent students on any selected date.
   - One-click CSV export with dynamic timestamps.
   - Complete persistence in SQLite across application restarts.

4. **Computer Vision Lab**:
   Interactive testing playground implementing 7 core digital image processing operations:
   - **Grayscale Conversion**: \(Y = 0.299R + 0.587G + 0.114B\)
   - **Brightness & Contrast**: Linear point operation \(g(x,y) = \alpha f(x,y) + \beta\)
   - **Histogram Equalization**: Dynamic CDF linearization with side-by-side Matplotlib frequency histograms.
   - **Spatial Smoothing**: Gaussian vs. Non-linear Median filter.
   - **Canny Edge Detection**: 4-stage pipeline with tunable low/high hysteresis thresholds.
   - **Morphological Operations**: Erosion, Dilation, Opening, Closing with rectangular, cross, and elliptical kernels.
   - **Image Thresholding**: Binary, Inverted, Truncate, Tozero, and Otsu's automated thresholding.

5. **Help & Viva Guide**:
   - Architecture workflows and file-by-file explanations.
   - Detailed mapping for all 10 Semester VII syllabus units (*Implemented*, *Explained conceptually*, *Not implemented*).
   - Database schema and ER relationships.
   - **28 Viva Examination Questions & Answers** matching the actual codebase.

---

## 🛠️ Technology Stack

- **Python 3.11+**: Primary programming language.
- **Streamlit**: Web frontend and interactive dashboard.
- **OpenCV (`cv2`)**: Computer vision pipeline and DNN inference engine.
- **YuNet ONNX**: Ultra-lightweight face detector (~232 KB) from OpenCV Zoo.
- **SFace ONNX**: Deep face recognition ResNet model (~38.6 MB) from OpenCV Zoo.
- **SQLite3**: Zero-configuration transactional SQL database.
- **NumPy & Pandas**: Numerical matrices, vector dot products, and tabular reports.
- **Matplotlib**: Intensity distribution histograms.

---

## 📁 Project Structure

```
smart_attendance/
│
├── app.py                          # Streamlit application with multi-page sidebar
├── database.py                     # SQLite schema, connection management & queries
├── face_recognition_service.py     # OpenCV YuNet detector & SFace recognizer wrapper
├── attendance_service.py           # Attendance business logic, duplicate checks & reports
├── image_processing.py             # 7 Computer Vision lab operations
├── requirements.txt                # Python dependencies
├── .env.example                    # Configuration template
├── README.md                       # Complete project documentation
│
├── models/                         # Pretrained ONNX models & download utility
│   ├── download_models.py          # Automatic downloader from official OpenCV Zoo
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── data/                           # Data storage & test samples
│   ├── attendance.db               # SQLite database file (auto-generated)
│   └── samples/                    # Sample face portraits & CV lab test image
│       ├── sample_student_1.jpg
│       ├── sample_student_2.jpg
│       └── cv_lab_sample.jpg
│
├── tests/                          # Automated unit test suite
│   ├── test_database.py
│   ├── test_face_recognition.py
│   ├── test_attendance.py
│   └── test_image_processing.py
│
└── docs/                           # Documentation & Viva references
    ├── viva_guide.md               # Detailed viva study guide & 28 Q&As
    └── syllabus_mapping.md         # Semester VII syllabus unit mapping
```

---

## 🚀 Installation & Execution (Windows)

### Step 1: Open PowerShell
Open Windows PowerShell and navigate to the project directory:
```powershell
cd d:\CV_Mini_Project_Sem_7
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Using Python 3.11
py -3.11 -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy warning, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Verify or Download Pretrained ONNX Models
The models are downloaded automatically by running:
```powershell
python models/download_models.py
```
This downloads `face_detection_yunet_2023mar.onnx` (232 KB) and `face_recognition_sface_2021dec.onnx` (38.6 MB) from the official OpenCV Zoo repository.

### Step 5: Run Automated Tests
Verify all 19 unit tests pass:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Streamlit will launch and display local access URLs:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```
Open `http://localhost:8501` in your web browser.

---

## 🔧 Camera Permission & Troubleshooting

1. **Webcam Not Opening**:
   - In Windows 10/11: Go to **Settings > Privacy & Security > Camera** and ensure **"Allow apps to access your camera"** and **"Allow desktop apps to access your camera"** are both turned **ON**.
   - In your browser (Chrome/Edge): Click the camera icon in the address bar and select **"Always allow http://localhost:8501 to access your camera"**.
2. **Viva / No Webcam Fallback**:
   - If a physical webcam is unavailable during presentation, both the **Register Student** and **Mark Attendance** pages have built-in tabs to **"Upload Image"** or **"Use Sample Test Image"** (bundled in `data/samples/`). You can demonstrate complete end-to-end face recognition with one click!
3. **Database Reset**:
   - To start with a fresh database, simply delete `data/attendance.db`. The application re-creates empty tables automatically upon restart.

---

## 📊 Syllabus Mapping (Semester VII)

| Unit | Syllabus Topic | Status |
| :--- | :--- | :--- |
| **Unit 1** | Introduction to Computer Vision (Formation, Camera, Resolution) | **Implemented** |
| **Unit 2** | Digital Image Processing (Color spaces, Grayscale conversion) | **Implemented** |
| **Unit 3** | Image Enhancement & Morphology (Contrast, HistEq, Morphology) | **Implemented** |
| **Unit 4** | Image Filtering & Edge Detection (Gaussian, Median, Canny) | **Implemented** |
| **Unit 5** | Feature Descriptors (5 Facial landmarks, 128-d Embeddings vs HOG) | **Implemented** |
| **Unit 6** | Image Segmentation (Binary, Truncate, Tozero, Otsu's method) | **Implemented** |
| **Unit 7** | CNNs & Pretrained Backbones (YuNet & SFace ONNX inference) | **Implemented** |
| **Unit 8** | Advanced CNN Techniques (Transfer learning, Attention mechanisms) | **Explained conceptually** |
| **Unit 9** | Generative Adversarial Networks (Generators, Discriminators) | **Explained conceptually** |
| **Unit 10** | Motion Detection & Video Processing (Frame capture & overlays) | **Implemented** |

*See `docs/syllabus_mapping.md` and `docs/viva_guide.md` for full theoretical breakdowns.*

---

## ⚠️ Limitations & Future Work

- **Single Session per Day**: Designed for single daily attendance per class. Future versions can add period/lecture timetables.
- **Lighting Sensitivity**: Deep recognition operates best in visible indoor/outdoor lighting. Infrared cameras can enhance low-light accuracy.
- **Anti-Spoofing (Liveness Detection)**: Current implementation uses static 2D images. 3D depth sensors or blink detection can be added in future versions.
