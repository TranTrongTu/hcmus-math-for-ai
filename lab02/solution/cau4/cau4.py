import time 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA as SklearnPCA

from lab02.solution.cau2.cau2 import CustomPCA 
def compare_visualizations(Z_custom, Z_lib, y) -> None:
    """
    Plot side-by-side comparison of Custom PCA vs Sklearn PCA projections.

    Args:
        Z_custom (np.ndarray): Projected data from CustomPCA, shape (2, n).
        Z_lib (np.ndarray): Projected data from Sklearn PCA, shape (n, 2).
        y (np.ndarray): Target labels for coloring the data points, shape (n,) 
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5), constrained_layout=True)
    
    # plot custom pca 
    scatter1 = ax1.scatter(Z_custom[0, :], Z_custom[1, :], c=y, 
                           cmap = "viridis", edgecolors='k', alpha = 0.7)
    
    ax1.set_title ("1. Custom PCA")
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.grid(True, linestyle = '--', alpha = 0.5)
    
    
    # plot library PCA 
    scatter2 = ax2.scatter(Z_lib[:, 0], Z_lib[:, 1], c=y, 
                           cmap = "viridis", edgecolors='k', alpha = 0.7)
    
    ax2.set_title ("2. Library PCA")
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.grid(True, linestyle = '--', alpha = 0.5)
    
    plt.colorbar(scatter1, ax=[ax1, ax2], label='Iris Classes')
    plt.suptitle('So sánh không gian giảm chiều (2D) của bộ Iris', fontsize=14, fontweight='bold')

    plt.savefig('cau4_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_pca_performance (X_sklearn: np.ndarray, y: np.ndarray, k: int) -> None:
    """
    Run both Custom and Sklearn PCA, visualize results, and compare metrics.
    
    Args:
        X_sklearn (np.ndarray): Original data matrix from sklearn, shape (n, m).
        y (np.ndarray): Target labels, shape (n,).
        k (int): Number of target dimensions.
    """
    # data for customPCA 
    X_custom = X_sklearn.T 
    m, n = X_custom.shape 
    
    start_time_custom = time.perf_counter()
    model_custom = CustomPCA(k)
    model_custom.fit(X_custom)
    Z_custom = model_custom.transform(X_custom)
    time_custom = time.perf_counter() - start_time_custom
    
    # tỷ lệ phương sai giữ lại(Explained Variance Ratio)
    # tỷ lệ: trị riêng thứ i / tổng tất cả trị riêng 
    total_variance_custom = np.sum(model_custom.eigenvalues)
    explained_variance_ratio_custom = model_custom.eigenvalues[:k] / total_variance_custom
    
    start_time_lib = time.perf_counter() 
    model_lib = SklearnPCA(n_components=k)
    Z_lib = model_lib.fit_transform(X_sklearn) # input is (n samples, m features)
    time_lib = time.perf_counter() - start_time_lib
    
    compare_visualizations(Z_custom, Z_lib, y)
    
    print(f"TRỊ RIÊNG (Eigenvalues / Explained Variance):")
    print(f" - Custom PCA : {np.round(model_custom.eigenvalues[:k], 2)}")
    print(f" - Sklearn PCA: {np.round(model_lib.explained_variance_, 2)}")
    
    print(f"TỶ LỆ PHƯƠNG SAI GIỮ LẠI (Explained Variance Ratio):")
    print(f" - Custom PCA : {np.round(explained_variance_ratio_custom, 2)}")
    print(f" - Sklearn PCA: {np.round(model_lib.explained_variance_ratio_, 2)}")
    
    # Tính toán sai số của các điểm dữ liệu sau khi được giảm chiều ở 2 ma trận giảm chiều dữ liệu
    Z_custom_aligned = Z_custom.T # match Z_lib shape 
    # Xử lý vấn đề Ngược dấu (Sign Flip) trong Đại số tuyến tính:
    # Lấy trị tuyệt đối của cả 2 ma trận trước khi trừ để loại bỏ sự khác biệt về chiều vector riêng
    
    absolute_difference = np.abs(np.abs(Z_custom_aligned) - np.abs(Z_lib))
    mean_absolute_error = np.mean(absolute_difference)
    
    print("SAI SỐ GIỮA 2 CÁCH (Mean Absolute Error):")
    print(f" - MAE: {mean_absolute_error:.16f}")
    print(f"\nTỐC ĐỘ CHẠY (Execution Time):")
    print(f" - Custom PCA: {time_custom:.6f} seconds")
    print(f" - Sklearn PCA: {time_lib:.6f} seconds")
    
if __name__ == '__main__':
    iris = datasets.load_iris()
    X_sklearn = iris.data # 150 samples, 4 features
    y = iris.target
    k = 2 
    evaluate_pca_performance(X_sklearn, y, k)
    
    
    