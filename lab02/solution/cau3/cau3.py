import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Import CustomPCA from your cau2.py file
from lab02.solution.cau2.cau2 import CustomPCA

def plot_original_and_pcs(X, mean_vector, eigenvectors, eigenvalues, y) -> None: 
    """
    Plot original 2D data and the Principal Components (eigenvectors)

    Args:
        X (np.ndarray): The original data matrix, shape (2, n)
        mean_vector (np.ndarray): The mean vector of the data, shape (2, 1)
        eigenvectors (np.ndarray): Matrix of eigenvectors, shape (2, 2)
        eigenvalues (np.ndarray): Array of eigenvalues, shape (2,)
        y (np.ndarray): Target labels for coloring the data points, shape (n,)
    """

    plt.figure(figsize=(8, 6))
    feature_1 = X[0, :] # x-axis 
    feature_2 = X[1, :] # y_axis
    scatter = plt.scatter(feature_1, feature_2, c=y, cmap='viridis', alpha=0.7, edgecolors='k')    
    # Explain: 
    # c=y: each point that has the same y is colored exactly same. 
    # cmap='viridis' just a colormap 
    # alpha = 0.7 show the intensity 
    plt.colorbar(scatter, label='Các lớp Iris (0, 1, 2)')
    origin_x = mean_vector[0, 0] # just hard code index because know shape X 
    origin_y = mean_vector[1, 0]
    
    # draw Principal Components
    for i in range(eigenvectors.shape[1]): 
        vector = eigenvectors[:, i] 
        length = np.sqrt(eigenvalues[i]) # find std 
        display_length = length * 1.5 # for easy visualization 
        
        # 1. Vẽ mũi tên bằng display_length
        plt.arrow(origin_x, origin_y, vector[0] * display_length, vector[1] * display_length, 
                  zorder = 5, head_width=0.15, head_length=0.2, fc='red', ec='red', 
                  linewidth=2)
        
        # 2. Tính tọa độ mũi nhọn của mũi tên
        tip_x = origin_x + vector[0] * display_length
        tip_y = origin_y + vector[1] * display_length
        
        # 3. Cộng thêm một khoảng offset (0.3) DỌC THEO vector để gắn chữ
        offset = 0.3
        
        plt.text(tip_x + vector[0] * offset, 
                 tip_y + vector[1] * offset, 
                 f'PC{i+1}', color='red', fontsize=12, fontweight='bold', 
                 ha='center', va='center', 
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))
        # show PC1 or PC2
    
    plt.title('Biểu đồ 2D Iris và các Principal Components')
    plt.xlabel('Sepal Length (Feature 1)')
    plt.ylabel('Sepal Width (Feature 2)')        
    
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig('cau3_plot1_original_data_with_pcs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_reduced_data(Z, y) -> None: 
    """ 
    Plot the data after reducing it to 1D space
    Args:
        Z (np.ndarray): The projected data matrix, shape (1, n)
        y (np.ndarray): Target labels for coloring the data points, shape (n,)
    """
    plt.figure(figsize = (8, 3))
    pc = Z[0, :]
    y_zeros = np.zeros_like(pc) # y = 0 universal x 
    
    scatter = plt.scatter(pc, y_zeros, c = y, cmap='viridis', alpha = 0.7, edgecolors='k')
    plt.colorbar(scatter, label='Các lớp Iris')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.yticks([]) 
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.savefig('cau3_plot2_reduced_1d_data.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__': 
    iris = datasets.load_iris() # 
    X_sklearn = iris.data[:, :2] # original 2D, before slicing: 150 samples, 4 features
    y = iris.target
    X = X_sklearn.T 
    m, n = X.shape
    k = 1 # reduced to 1D 
    model = CustomPCA(k)
    model.fit(X)
    Z = model.transform(X)
    plot_original_and_pcs(X, model.mean_vector, model.eigenvectors, model.eigenvalues, y)
    plot_reduced_data(Z, y)
    