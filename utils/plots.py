import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_2d_clusters(X_2d, predicted_labels, true_labels=None, title='2D Projection', save_path=None):
    if true_labels is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=predicted_labels, 
                            cmap='viridis', alpha=0.6, s=30)
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.set_title(f'{title} - Predicted Clusters')
        plt.colorbar(scatter1, ax=ax1)
        
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels, 
                            cmap='coolwarm', alpha=0.6, s=30)
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        ax2.set_title(f'{title} - True Labels')
        plt.colorbar(scatter2, ax=ax2)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=predicted_labels, 
                        cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_elbow_curve(k_values, inertias, optimal_k=None, save_path=None):
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, 'o-', linewidth=2, markersize=8)
    
    if optimal_k:
        plt.axvline(optimal_k, color='red', linestyle='--', 
                label=f'Optimal k={optimal_k}')
        plt.legend()
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_bic_aic_curves(k_values, bic_dict, aic_dict, save_path=None):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for cov_type in bic_dict.keys():
        ax1.plot(k_values, bic_dict[cov_type], marker='o', label=cov_type)
        ax2.plot(k_values, aic_dict[cov_type], marker='s', label=cov_type)
    
    ax1.set_xlabel('Number of Components (k)')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC Scores (lower is better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    ax2.set_xlabel('Number of Components (k)')
    ax2.set_ylabel('AIC')
    ax2.set_title('AIC Scores (lower is better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(train_loss, val_loss=None, save_path=None):
    """
    Plot autoencoder training curves (loss vs epochs).
    """
    plt.figure(figsize=(8, 5))
    
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    
    if val_loss is not None and len(val_loss) > 0:
        plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_comparison_heatmap(data, row_labels, col_labels, title='Methods Comparison', save_path=None):
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Experiments')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    
    if class_names:
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()