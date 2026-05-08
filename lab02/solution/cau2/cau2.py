import numpy as np 

class CustomPCA: 
    """
    Principal Component Analysis (PCA) algorithm implemented from scratch.
    It expects the input data matrix X to be in the shape of (m, n), 
    where m is the number of features and n is the number of samples.
    """
    def __init__(self, k: int):
        """
        Initialize the PCA model.
        
        Args:
            k (int): Target dimensions (number of principal components to keep). 
                     Must be less than or equal to m.
        
        Attributes:
            self.k (int): Target dimensions.
            self.mean_vector (np.ndarray): The mean vector of the dataset, shape (m, 1).
            self.eigenvalues (np.ndarray): Sorted eigenvalues of the covariance matrix, shape (m,).
            self.eigenvectors (np.ndarray): Sorted eigenvectors, shape (m, m).
            self.W (np.ndarray): Projection matrix containing top k eigenvectors, shape (m, k).
        """
        self.k = k
        self.mean_vector = None
        self.eigenvalues = None 
        self.eigenvectors = None 
        self.W = None 
    
    def fit(self, X: np.ndarray) -> None:
        """
        Step 1 -> 4: Learn the principal components from the data 
        Args:
        X: Data matrix of shape (m, n) -> m features, n samples.
        """
        m, n = X.shape
        # step 1: normalizing data
        self.mean_vector = np.mean(X, axis = 1, keepdims= True) # keepdims to avoid flattening to 1D
        X_hat = X - self.mean_vector 
        
        # step 2: calculate covariance matrix S 
        S = np.dot (X_hat, X_hat.T) / (n - 1)
        
        # step 3: find eigenvalues and eigenvectors 
        eigenvalues, eigenvectors = np.linalg.eigh(S) # return ascending order
        
        # Step 4: Sort and select top k principal components
        self.eigenvalues = eigenvalues[::-1]
        self.eigenvectors = eigenvectors[:, ::-1]
        self.W = self.eigenvectors[:, :self.k]
        
    def transform (self, X: np.ndarray) -> np.ndarray: 
        """
        Step 5: Project data onto the new k-dimensional space. 
        Args: 
            X(np.ndarray): data matrix (m, n) to be transformed. 
        
        Returns: 
        np.ndarray: Transformed data matrix of shape (k, n).
        """
        if self.W is None: # X must be fited before transforming  
            raise ValueError("The PCA model must be fitted before transforming data.")
        X_hat = X - self.mean_vector 
        # W ^ T @ X_hat 
        res = np.dot(self.W.T, X_hat)
        return res 
    
    
if __name__ == "__main__": 
    np.random.seed(42)
    m, n, k = 5, 10, 2 
    X = np.random.randint(0, 20, size = (m, n)).astype(float) 
    
    print(f"MA TRẬN DỮ LIỆU GỐC X (m = {m} đặc trưng, n = {n} mẫu):")
    print(X)
    print()
    # Training 
    model = CustomPCA(k = k)
    model.fit(X)
    print("Yêu cầu 3:")
    print(f"{m} trị riêng của ma trận hiệp phương sai:")
    print(np.round(model.eigenvalues, 2))
    print() 
    print(f"{m} các vector riêng tương ứng với {m} trị riêng:") 
    print(np.round(model.eigenvectors, 2))
    
    print()
    print("Yêu cầu 4:")
    print(f"{k} vector thành phần chính:")
    print(np.round(model.W, 2))
    
    print() 
    print("Yêu cầu 5:")
    vector_input = np.random.randint(0, 20, (m, 1)).astype(float)
    print("vector ban đầu:")
    print(vector_input)
    vector_projection = model.transform(vector_input)
    print("vector sau khi chiếu:")
    print(np.round(vector_projection, 2))    