# Semester VII Computer Vision Syllabus Mapping

This document provides the formal mapping between the **Semester VII Computer Vision Syllabus** (Units 1 to 10 and practical curriculum) and the **Smart Attendance Management System Using Face Recognition**.

Each syllabus topic is categorized strictly under one of three statuses:
- **Implemented**: Executed by functional Python and OpenCV code in the application.
- **Explained conceptually**: Relevant theoretical concept explained in the Help & Viva Guide for viva examination.
- **Not implemented**: Advanced, heavy, or out-of-scope technique omitted to keep the project lightweight, modular, and reliable.

---

## 1. Ten Syllabus Units Mapping

| Unit | Syllabus Topic | Status | Implementation Details in This Project |
| :--- | :--- | :--- | :--- |
| **Unit 1** | **Introduction to Computer Vision**<br>• Image representation (matrices, channels)<br>• Image formation & camera models<br>• Spatial resolution & quantization | **Implemented** | • BGR/Grayscale NumPy array representation (`H × W × C`, `uint8`).<br>• Camera frame acquisition via Streamlit `st.camera_input` and OpenCV video stream.<br>• Dimension & resolution validation (`validate_image`, minimum face bounding box checks). |
| **Unit 2** | **Digital Image Processing**<br>• Image loading, color spaces (BGR, RGB, YCrCb)<br>• Grayscale conversion<br>• Point transformations | **Implemented** | • Image reading & encoding via OpenCV (`cv2.imread`, `cv2.imdecode`).<br>• BGR to Grayscale conversion using luminance weighting (\(Y = 0.299R + 0.587G + 0.114B\)) in `image_processing.to_grayscale()`.<br>• Color space conversion to YCrCb for luminance equalization. |
| **Unit 3** | **Image Enhancement & Morphology**<br>• Brightness & contrast adjustment<br>• Histogram calculation & equalization<br>• Morphological erosion, dilation, opening, closing | **Implemented** | • Linear transformation \(g(x,y) = \alpha f(x,y) + \beta\) in `image_processing.adjust_brightness_contrast()`.<br>• Global contrast enhancement via `cv2.equalizeHist()` with Matplotlib side-by-side histogram plots.<br>• Morphological filtering with rectangular, cross, and elliptical structuring elements in `image_processing.apply_morphology()`. |
| **Unit 4** | **Image Filtering and Edge Detection**<br>• Neighborhood spatial filtering<br>• Gaussian & median smoothing<br>• Canny edge detection | **Implemented** | • Spatial convolution using 2D Gaussian kernel (`cv2.GaussianBlur`) for Gaussian noise reduction.<br>• Non-linear median filter (`cv2.medianBlur`) for impulse (salt-and-pepper) noise in `image_processing.apply_smoothing()`.<br>• Multi-stage Canny edge detection (Gaussian blur, Sobel gradient, non-maximum suppression, hysteresis thresholding) in `image_processing.apply_canny()`. |
| **Unit 5** | **Feature Detection & Descriptors**<br>• Traditional feature descriptors (HOG, SIFT, ORB)<br>• Deep face embeddings & landmark descriptors<br>• Metric learning & cosine distance | **Implemented** | • 5-point facial landmark detection (eyes, nose, mouth corners) via YuNet.<br>• 128-dimensional deep feature embedding via OpenCV SFace CNN in `face_recognition_service.py`.<br>• Cosine similarity metric matching against registered templates in SQLite.<br>• *Theoretical comparison with traditional HOG features provided in Viva Guide.* |
| **Unit 6** | **Image Segmentation**<br>• Global thresholding (Binary, Inverted, Truncate, Tozero)<br>• Otsu's optimal automatic thresholding<br>• Advanced segmentation (Watershed, GrabCut) | **Implemented** (Basic & Otsu)<br>*(Watershed/GrabCut: Explained conceptually)* | • Global thresholding operations via `cv2.threshold()` in `image_processing.apply_thresholding()`.<br>• Automatic inter-class variance maximization via Otsu's thresholding (`cv2.THRESH_OTSU`).<br>• GrabCut/Watershed explained conceptually in viva guide. |
| **Unit 7** | **CNNs and Image Classification**<br>• Convolutional layers, activation functions, pooling<br>• Deep feature representations<br>• Pretrained vision backbones | **Implemented** (Inference via ONNX)<br>*(Training: Explained conceptually)* | • Pretrained CNN inference using OpenCV DNN module (`cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`).<br>• Deep representations extracted without local GPU training overhead.<br>• CNN architecture (Conv2D, BatchNorm, ReLU, Residual blocks, Global Pooling) explained in Viva Guide. |
| **Unit 8** | **Advanced CNN Techniques**<br>• Transfer learning & fine-tuning<br>• Attention mechanisms<br>• Modern vision transformers | **Explained conceptually** | • Theoretical explanation of transfer learning: why we use pretrained SFace embeddings rather than training a CNN from scratch on a small college dataset.<br>• Attention mechanism concepts explained in Viva Guide.<br>• Not trained locally to preserve lightweight mini-project scope. |
| **Unit 9** | **Generative Adversarial Networks (GANs)**<br>• Generator & discriminator architectures<br>• Minimax game loss formulation<br>• Vision data augmentation | **Explained conceptually** | • Theoretical breakdown of GANs in Viva Guide.<br>• Role of synthetic face generation (e.g. StyleGAN) and data augmentation discussed.<br>• Not implemented in code (out of scope for attendance tracking). |
| **Unit 10** | **Motion Detection & Video Processing**<br>• Frame-by-frame webcam capture<br>• Frame difference & optical flow<br>• Real-time video analytics | **Implemented** (Webcam & frame processing)<br>*(Optical flow: Explained conceptually)* | • Live frame capture and processing in Streamlit (`st.camera_input` and OpenCV frame decoding).<br>• Real-time face detection, alignment, recognition, and visual bounding box overlays.<br>• Optical flow and background subtraction explained conceptually in Viva Guide. |

---

## 2. Practical Curriculum Mapping

| Practical Name | Application Module | Status |
| :--- | :--- | :--- |
| **1. Image Loading, Display & Grayscale Conversion** | `image_processing.to_grayscale()` | **Implemented** |
| **2. Brightness and Contrast Enhancement** | `image_processing.adjust_brightness_contrast()` | **Implemented** |
| **3. Histogram Calculation & Histogram Equalization** | `image_processing.equalize_histogram()`, `plot_histogram_comparison()` | **Implemented** |
| **4. Spatial Smoothing (Gaussian & Median Filters)** | `image_processing.apply_smoothing()` | **Implemented** |
| **5. Canny Edge Detection** | `image_processing.apply_canny()` | **Implemented** |
| **6. Morphological Operations (Erosion, Dilation, Opening, Closing)** | `image_processing.apply_morphology()` | **Implemented** |
| **7. Image Thresholding (Global & Otsu's Method)** | `image_processing.apply_thresholding()` | **Implemented** |
| **8. Face Detection and Facial Landmark Extraction** | `face_recognition_service.py` (OpenCV YuNet) | **Implemented** |
| **9. Face Recognition & Feature Matching** | `face_recognition_service.py` (OpenCV SFace) | **Implemented** |
| **10. Optical Flow & Motion Tracking** | Help & Viva Guide (Section 6 & 11) | **Explained conceptually** |
| **11. Object Detection (YOLO / SSD)** | Help & Viva Guide (Section 11 & Q18) | **Explained conceptually** |
