import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from lab02.solution.cau2.cau2 import CustomPCA 
def compare_dimensionality_reduction() -> None:
    # 1. Load the Digits dataset (8x8 images of handwritten digits)
    # X shape: (1797, 64) - 1797 samples, 64 features (pixels)
    # y shape: (1797,) - labels from 0 to 9
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    target_names = digits.target_names

    print(f"Original dataset shape: {X.shape}")

    # 2. Initialize the models (reducing to 2D for visualization)
    pca = CustomPCA(2)
    lda = LDA(n_components=2)
    tsne = TSNE(n_components=2, init='pca', random_state=42)

    # 3. Fit and transform the data
    pca.fit(X.T)
    X_pca = pca.transform(X.T).T

    X_lda = lda.fit_transform(X, y)

    X_tsne = tsne.fit_transform(X)

    # 4. Visualization Setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.get_cmap('tab10', 10) # Get 10 distinct colors for 10 digits

    models = [
        ("PCA (Unsupervised - Variance)", X_pca),
        ("LDA (Supervised - Class Separation)", X_lda),
        ("t-SNE (Unsupervised - Neighborhood)", X_tsne)
    ]

    # 5. Plotting loop
    for i, (title, X_transformed) in enumerate(models):
        ax = axes[i]
        for color_idx, target_name in zip(range(10), target_names):
            # Scatter plot for each digit class
            ax.scatter(X_transformed[y == color_idx, 0], 
                       X_transformed[y == color_idx, 1], 
                       color=colors(color_idx), alpha=0.6, label=target_name, s=15)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Add a single legend for all plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=10, title="Digit Classes")
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for legend at the bottom
    plt.savefig("pca_vs_lda_vs_tsne.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    compare_dimensionality_reduction()