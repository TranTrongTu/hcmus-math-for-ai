import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, load_digits
from lab02.solution.cau2.cau2 import CustomPCA
from cau5 import CompressiblePCA
class AdvancedPCA(CustomPCA): 
    def __init__(self, k):
        """
        Initialize AdvancedPCA by inheriting from CustomPCA.
        """
        super().__init__(k) # Call the constructor of the parent class
        
    def transform_whitened(self, X: np.ndarray) ->np.ndarray:
        """
        Project data onto the new k-dimensional space and apply whitening.
        Whitening ensures that all principal components have a variance of 1.
        
        Formula: Z_whitened = Z / sqrt(lambda)
        
        Args: 
        X: data matrix (m, n) to be transformed. 
        
        Returns: 
        Whitened transformed data matrix: (k, n)
        """
        # Step 1: Get the standard projection Z of shape (k, n) from parent class
        Z = super().transform(X)
        
        # Step 2: Get the top k eigenvalues corresponding to the k principal components
        top_k_eigenvalues = self.eigenvalues[:self.k]
        
        # Step 3: Calculate the standard deviation for each component
        # Reshape to (k, 1) to allow row-wise broadcasting across the n samples
        std_devs = np.sqrt(top_k_eigenvalues).reshape(-1, 1)
        
        # Step 4: whitening 
        Z_whitened = Z / std_devs
        
        return Z_whitened


def demo_eigenfaces() -> None:
    # Loading 400 facial images (64x64 pixel)
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    faces = dataset.data
    image_shape = (64, 64) # for reshape later 

    X = faces.T 
    pca = AdvancedPCA(k = 10)
    pca.fit(X)
    
    # Note: Z_whitened is prepared here for downstream classification tasks (e.g., KNN, SVM).
    # It is NOT used for drawing Eigenfaces (which relies strictly on the eigenvectors W).
    Z_whitened = pca.transform_whitened(X)    
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Ứng dụng 1: Nhận dạng khuôn mặt (Eigenfaces)", fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flatten()):
        # Reshape each eigenvector into a 64x64 square image 
        eigenface = pca.W[:, i].reshape(image_shape)
        ax.imshow(eigenface, cmap='gray')
        ax.set_title(f"Eigenface {i+1}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("demo_eigenfaces.png", dpi=300)
    plt.show()


def demo_denoising():
    """Demonstrate the signal denoising capability of PCA on the Digits dataset."""
    # Load a random digit image 
    digits = load_digits()
    clean_images = digits.data
    
    # Create Gaussian Noise 
    np.random.seed(42)
    noisy_images = clean_images + np.random.normal(0, 4, clean_images.shape) # 4 is large 

    X_input = noisy_images.T 

    my_pca = CompressiblePCA(k=12)
    my_pca.fit(X_input)
    Z = my_pca.transform(X_input)

    
    filtered_images_T = my_pca.inverse_transform(Z)
    filtered_images = filtered_images_T.T

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Ứng dụng 2: Giảm nhiễu tín hiệu", fontsize=16, fontweight='bold')
    
    axes[0].imshow(clean_images[0].reshape(8, 8), cmap='gray')
    axes[0].set_title("1. Ảnh gốc ban đầu")
    axes[0].axis('off')
    
    axes[1].imshow(noisy_images[0].reshape(8, 8), cmap='gray')
    axes[1].set_title("2. Ảnh bị nhiễu (Thêm Noise)")
    axes[1].axis('off')
    
    axes[2].imshow(filtered_images[0].reshape(8, 8), cmap='gray')
    axes[2].set_title("3. Ảnh sau khi lọc qua PCA")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("demo_denoising.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    demo_eigenfaces()
    demo_denoising()