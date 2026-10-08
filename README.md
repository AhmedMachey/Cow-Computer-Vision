# Cow-Computer-Vision
# Image Segmentation, Weight Estimation & Cattle Classification

<p align="center">
  <strong>Computer Vision • Image Processing • Smart Agriculture</strong>
</p>

<p align="center">
  A Python-based computer vision project for analyzing cattle images, extracting their silhouette and morphological features, estimating their weight, and classifying them as dairy or beef cattle.
</p>

---

## 📌 Overview

This project explores the use of **image processing and computer vision techniques** for automated cattle analysis.

Starting from a cattle image, the project includes several processing stages:

- Image preprocessing
- Image segmentation
- Binary mask generation
- Morphological processing
- Contour extraction
- Morphological measurements
- Weight estimation
- Dairy / beef classification
- Result visualization

The repository also contains several output directories used to store intermediate and final processing results.

> **Project context:** Engineering project — ING_2, University of Monastir, academic year 2025/2026.

---

## 🎯 Main Objectives

- 🐄 Automatically segment the cattle from its background.
- 📐 Extract morphological characteristics from the animal's silhouette.
- 📊 Calculate the projected surface area in pixels.
- ⚖️ Estimate the cattle weight using a calibrated relationship.
- 🥛 Classify the animal as **dairy cattle** or **beef cattle**.
- 🖼️ Generate visual outputs that make the results easy to interpret.

---

# 🔬 Processing Pipeline

```text
                    INPUT IMAGE
                         │
                         ▼
               ┌─────────────────┐
               │  Preprocessing  │
               └─────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │   Segmentation  │
               └─────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │  Binary Mask    │
               └─────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │  Morphological  │
               │   Processing    │
               └─────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │    Contours     │
               └─────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │ Measurements    │
               │ Surface/Height  │
               │ /Width          │
               └─────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │ Weight Estimate │
               └─────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │ Classification  │
               │ Dairy / Beef    │
               └─────────────────┘
                         │
                         ▼
                FINAL VISUALIZATION
```

---

# 🧠 Methodology

## 1. Image Preprocessing

The preprocessing stage prepares the input image for segmentation.

The project uses:

- Grayscale conversion
- Gaussian filtering
- CLAHE contrast enhancement
- Normalization

---

## 2. Image Segmentation

The main segmentation workflow uses the **GrabCut algorithm** to separate the cattle from the background.

The segmentation process produces a foreground mask in which the animal can be isolated from the surrounding scene.

---

## 3. Binary Mask & Morphological Processing

After segmentation, a binary representation of the cattle is generated.

Morphological operations are then used to clean and improve the mask:

- **Closing** to reduce small holes and discontinuities.
- **Opening** to remove small external noise.

---

## 4. Contour Extraction

Contours are detected from the processed binary mask.

The **main contour** is selected using the largest detected area so that the analysis focuses on the cattle rather than small objects or noise.

---

## 5. Morphological Measurements

The project extracts geometric information from the main contour and its bounding rectangle.

| Measurement | Description |
|---|---|
| Projected Surface | Area of the main cattle contour in pixels |
| Height | Height of the bounding rectangle |
| Width | Width of the bounding rectangle |
| Silhouette | Visible shape of the segmented cattle |

---

# ⚖️ Weight Estimation

The project uses a calibrated linear relationship between the segmented surface and the estimated cattle weight:

```text
Weight (kg) = a × Surface (px) + b
```

Where:

- `a` = calibration coefficient
- `b` = corrective constant
- `Surface (px)` = measured projected area of the segmented cattle

> **Note:** This is an image-based estimation model and should be treated as an approximate estimate rather than a replacement for professional weighing equipment.

---

# 🐄 Cattle Classification

The project applies the following predefined threshold:

```text
Estimated Weight > 650 kg
        └── Beef Cattle

Estimated Weight ≤ 650 kg
        └── Dairy Cattle
```

---

# 🖼️ Output Results

The current repository contains several output folders:

```text
out/
out_auto/
out_multi/
out_multiple/
output_contours/
segmentation_output/
```

These folders can contain segmentation, contour, multi-image, and final processing results.

---

# 📂 Repository Structure

The project structure shown in the current project folder is:

```text
cattle-computer-vision/
│
├── src/
│
├── out/
├── out_auto/
├── out_multi/
├── out_multiple/
├── output_contours/
├── segmentation_output/
│
├── deeplab_v3.tflite
│
├── prj1.0.py
├── prj2.0.py
├── prj03.py
├── prjard.py
├── projet traitement de image.py
├── TEST.py
│
├── Untitled-1.ipynb
│
└── README.md
```

### File organization

| File / Folder | Role |
|---|---|
| `src/` | Source code and project modules |
| `out/` | Processing outputs |
| `out_auto/` | Automatic processing outputs |
| `out_multi/` | Multi-image processing outputs |
| `out_multiple/` | Multiple processing results |
| `output_contours/` | Contour extraction results |
| `segmentation_output/` | Segmentation results |
| `prj1.0.py` | Development version |
| `prj2.0.py` | Development version |
| `prj03.py` | Development version |
| `prjard.py` | Development version |
| `projet traitement de image.py` | Project image-processing script |
| `TEST.py` | Testing script |
| `Untitled-1.ipynb` | Jupyter Notebook |
| `deeplab_v3.tflite` | TensorFlow Lite model file present in the project |

> The exact role of each development script should be updated after choosing the final validated version.

---

# 🛠️ Technologies

- **Python**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Jupyter Notebook**
- **GrabCut**
- **CLAHE**
- **Morphological Image Processing**
- **Contour Analysis**

The repository also contains `deeplab_v3.tflite`. Its exact use in the final pipeline should be documented once confirmed from the source code.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cattle-computer-vision.git
cd cattle-computer-vision
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

If `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by the scripts:

```bash
pip install opencv-python numpy matplotlib jupyter
```

---

# ▶️ Usage

The repository currently contains several Python scripts and a Jupyter Notebook.

Example:

```bash
python "projet traitement de image.py"
```

Or launch the notebook:

```bash
jupyter notebook
```

Then open:

```text
Untitled-1.ipynb
```

> Use the final validated script for demonstrations. The other `prj*.py` files can be kept as development history or moved into an `archive/` folder.

---

# 📊 Expected Results

A successful run should produce visual results containing information such as:

- Cattle silhouette
- Main contour
- Bounding box
- Estimated weight
- Cattle classification

These results can be used for analysis, documentation, and livestock monitoring.

---

# 🌱 Applications

Potential applications include:

- Automated cattle monitoring
- Non-invasive weight estimation
- Herd management
- Morphological analysis
- Agricultural data collection
- Livestock production monitoring

---

# 🔮 Future Improvements

### 🎥 Real-Time Video
Extend the system from individual images to real-time video analysis.

### 🤖 AI-Based Segmentation
Integrate advanced segmentation models to improve robustness in complex environments.

### 📦 3D Body Analysis
Use multiple viewpoints or depth information to estimate cattle body volume.

### 🐄 Individual Recognition
Add cattle identification and long-term tracking.

### 📈 Historical Monitoring
Store measurements over time to analyze weight evolution and herd statistics.

### 🩺 Health-Oriented Analysis
Explore image-based indicators for animal health and condition monitoring.

---

# ⚠️ Limitations

- Weight estimation depends on the calibration parameters.
- Measurements are based on image pixels.
- Camera position and cattle pose may affect the results.
- Segmentation quality depends on image conditions.
- The 650 kg threshold is a predefined project criterion.

---

# 🎓 Academic Context

**Project Title:**  
*Segmentation d’Images pour l’Estimation du Poids et la Classification des Bovins (Laitiers / Viande)*

**Student:** Machey Ahmed  
**Level:** ING_2  
**Institution:** Institut Supérieur d'Informatique et de Mathématiques de Monastir  
**University:** University of Monastir  
**Academic Year:** 2025/2026  
**Supervisor:** Mme Afef Abidi

---

# 👨‍💻 Author

**Machey Ahmed**  
Engineering Student — University of Monastir, Tunisia

---

# ⭐ Acknowledgments

Special thanks to **Mme Afef Abidi** for her guidance, technical support, and constructive feedback throughout the project.

---

# 📜 License

This repository is intended primarily for **academic and educational purposes**.

An open-source license can be added according to the author's preferred distribution terms.

---

# 🔗 Project Links

**GitHub Repository:**  
`https://github.com/AhmedMachey/Cow-Computer-Vision

**Project Report:**  
Available in this repository.

---

<p align="center">
  🐄 <strong>Computer Vision × Smart Agriculture</strong> 🌱
</p>
