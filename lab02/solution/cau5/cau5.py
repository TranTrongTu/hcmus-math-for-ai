import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from lab02.solution.cau2.cau2 import CustomPCA 

class CompressiblePCA(CustomPCA):
    """
    Adding some functions from CustomPCA
    - Change flexibility k (do not require fit again)
    - Decompression Image (Inverse Transform)
    """
    def __init__(self, k):
        super().__init__(k) 
        
    def set_k(self, new_k: int) -> None:
        """Allow change reduction dimension k after fitting"""
        if self.eigenvectors is None:
            raise ValueError("Must call function fit() before changing reduction dimension k")
        self.k = new_k
        # Update projection matrix W with the new k columns
        self.W = self.eigenvectors[:, :self.k]

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Decompress: convert k dimension to original dimension m
        Formula: X_approx = W @ Z + Mean
        """
        if self.W is None or self.mean_vector is None:
            raise ValueError("Must call function fit() before back to original space.")
        
        
        X_approx = np.dot(self.W, Z) + self.mean_vector
        return X_approx
    

def custom_rgb2gray(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image (H, W, 3) to Grayscale (H, W) using Luminosity method.
    """
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]
    
    gray_matrix = 0.299 * R + 0.587 * G + 0.114 * B
    return gray_matrix

def preprocess_image(image_path: str): 
    """
    Load image, convert to Grayscale and cast to float matrix.
    
    Returns: 
    X (np.ndarray): Original image data matrix, shape (m, n)
    original_gray_image (np.ndarray): Original gray image, shape (m, n)
    """
    img_rgb = mpimg.imread(image_path)
    
    # Check if image is already gray 
    if len(img_rgb.shape) == 2:
        original_gray_image = img_rgb
    else:

        # Select the first 3 RGB channels
        if img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]

        original_gray_image = custom_rgb2gray(img_rgb)
        
    # Handle mpimg value range discrepancy (PNG: 0.0-1.0, JPG: 0-255)
    # Normalize to 0-255 if the image data is in the 0.0-1.0 range
    if original_gray_image.max() <= 1.0:
        original_gray_image = original_gray_image * 255.0

   
    X = original_gray_image.astype(np.float64)
    return X, original_gray_image

def calculate_mse(original_img: np.ndarray, reconstructed_img: np.ndarray) -> float:  
    """
    Args:
        original_img (np.ndarray): shape (m, n)
        reconstructed_img (np.ndarray): shape(m, n)
    Return:
    return Mean Square Error
    """
    
    mse = np.mean((original_img - reconstructed_img) ** 2)
    return mse 

def compress_and_visualize(image_path: str, k_values: list = [2, 5, 10, 20, 50, 100]) -> None:
    """
    Thực hành: (b) Áp dụng PCA, (c) Tái tạo ảnh, (d) Đánh giá, (e) Trực quan hóa
    Thực hiện trên một bức ảnh
    """
    # a. preprocess image 
    X, original_gray_image = preprocess_image(image_path)
    # find all eigenvalues because will choose k later 
    model = CompressiblePCA(k = X.shape[0]) 
    model.fit(X)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten() # using index from 0 to 7 instead of using axes[row, col]
    
    # first cell: display original image 
    axes[0].imshow(original_gray_image, cmap = 'gray')
    axes[0].set_title(f"Ảnh gốc \nShape: {original_gray_image.shape}")
    axes[0].axis('off')
    
    for i, k in enumerate(k_values): 
        model.set_k(k)
        Z = model.transform(X)        
        # Reconstruct image 
        X_approximate = model.inverse_transform(Z)
        
        mse_error = calculate_mse(original_gray_image, X_approximate)
        ax = axes[i + 1] 
        X_approx_clipped = np.clip(X_approximate, 0, 255) # we just see color in this range
        
        # Tỷ lệ nén (Sơ bộ)
        # Gốc: m * n. Nén: (m * k) cho ma trận W + (k * n) cho ma trận Z
        m, n = X.shape
        original_size = m * n
        compressed_size = (m * k) + (k * n)
        compression_ratio = original_size / compressed_size
        
        ax.imshow(X_approx_clipped, cmap='gray')
        ax.set_title(f"k = {k}\nMSE: {mse_error:.1f}\nTỷ lệ nén: {compression_ratio:.1f}x")        
        ax.axis('off')   
    axes[7].axis('off')
    
    plt.suptitle(f"Ứng dụng PCA Nén ảnh - {image_path}", fontsize=16, fontweight='bold')
    plt.tight_layout(h_pad=6.0, rect=[0, 0, 1, 0.93])
    plt.savefig(f"cau5_ketqua_{image_path.split('.')[0]}.png", dpi=300)
    plt.show()
    
if __name__ == "__main__":
    # Thay tên file bằng đúng tên ảnh bạn đã lưu
    compress_and_visualize("anh_cup.jpg")
    compress_and_visualize("anhcasau.jpg")