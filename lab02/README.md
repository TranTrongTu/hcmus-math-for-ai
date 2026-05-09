# 🌌 Lab 02: Dimensionality Reduction with PCA (Implementations from Scratch)

## 📝 Overview
This project is the second assignment for the **Mathematical Methods for Artificial Intelligence** course at **VNU-HCM University of Science (HCMUS)**. It focuses on the comprehensive implementation of **Principal Component Analysis (PCA)** from the ground up using pure `NumPy`. The project transitions from basic mathematical derivations to complex applications like image compression, signal denoising, and manifold learning comparison.

## ✨ Technical Highlights
- **Object-Oriented Implementation:** Built a modular hierarchy starting from `CustomPCA` to specialized classes like `CompressiblePCA` (for reconstruction) and `AdvancedPCA` (for whitening).
- **Core Mathematical Logic:** Manually implemented Mean-centering, Covariance Matrix calculation, and Eigenvalue Decomposition to extract Principal Components.
- **Image Compression & Denoising:** Applied PCA to reduce the storage size of high-resolution images and filtered Gaussian noise from the Digits dataset.
- **Whitening & Eigenfaces:** Extracted "Eigenfaces" from the Olivetti faces dataset and applied whitening to ensure uncorrelated features with unit variance.
- **Algorithm Benchmarking:** Conducted a three-way comparison between **PCA** (Unsupervised), **LDA** (Supervised), and **t-SNE** (Non-linear) to analyze class separation in latent spaces.

## 📂 Directory Structure

<details>
<summary><b>Click to view the detailed project workspace</b></summary>
<br>


```text
lab02/
├── De_lab_2.pdf             # Original assignment requirements
├── report.pdf               # Comprehensive mathematical and experimental report
└── solution/                # Core implementation and applications
    ├── cau2/
    │   └── cau2.py          # CustomPCA class: The core mathematical engine
    ├── cau3/
    │   ├── cau3.py          # Iris dataset projection (2D to 1D)
    │   ├── cau3_plot1_...png # Visualization: Original data with Principal Components
    │   └── cau3_plot2_...png # Visualization: Data projected onto 1D space
    ├── cau4/
    │   ├── cau4.py          # Benchmarking: Custom PCA vs. scikit-learn PCA (MAE ≈ 0)
    │   └── cau4_comparison_plot.png # Visual side-by-side comparison
    ├── cau5/
    │   ├── cau5.py          # Image compression using CompressiblePCA class
    │   ├── anh_cup.jpg      # Input test image (Cup)
    │   ├── anhcasau.jpg     # Input test image (Crocodile)
    │   └── cau5_ketqua_...png # Reconstructed images at various k-components
    └── nangcao/             # Advanced AI implementations
        ├── ung_dung_pca.py  # Whitening, Eigenfaces, and Signal Denoising
        ├── sosanhpca.py     # Comparative analysis: PCA vs. LDA vs. t-SNE
        ├── demo_denoising.png # Proof-of-concept for noise reduction
        ├── demo_eigenfaces.png # Visualization of Principal Components for faces
        └── pca_vs_lda_vs_tsne.png # Manifold learning comparison plot
```
</details>

## 🚀 Getting Started

### 1. Installation
Ensure you have Python installed. Install the necessary dependencies for matrix operations and visualization:
```bash
pip install -r requirements.txt
```

### 2. Execution
Each module can be run independently to verify specific tasks:
- **Core PCA Test:** `python solution/cau2/cau2.py`
- **Image Compression:** `python solution/cau5/cau5.py`
- **Algorithm Comparison:** `python solution/nangcao/sosanhpca.py`

## 📊 Experimental Results
- **Precision:** The `CustomPCA` implementation yields results identical to `scikit-learn` (Mean Absolute Error $\approx 10^{-15}$).
- **Efficiency:** Successfully reconstructed grayscale images with high fidelity using only the top 10-20% of principal components, demonstrating significant storage reduction.
- **Analytical Insights:** Proven that while PCA is excellent for variance preservation, LDA is superior for class separation, and t-SNE excels at capturing local non-linear structures for visualization.