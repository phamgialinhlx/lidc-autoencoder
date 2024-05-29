import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import numpy as np
from scipy.stats import entropy
from src.entropy2d import calcEntropy2d
import matplotlib.pyplot as plt
import os


DIR1 = "./feature_maps/downstream_unetr_soft_dice_v2"
DIR2 = "./feature_maps/swin_cls_seg_soft_dice_v2"
OUTPUT_DIR = "./entropy_histograms"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_entropy(tensor):
    histogram, _ = np.histogram(tensor, bins=1000, density=True)
    hist_nonzero = histogram[histogram > 0]  # Remove zero entries for entropy calculation
    ent = entropy(hist_nonzero)
    return ent

def tensor_histogram(tensor1, tensor2, entropy1, entropy2, bins=100, save_path=None):
    """
    Plot and save histogram of tensor values with entropy values displayed in the title.

    Parameters:
    - tensor1: NumPy array representing the first tensor
    - tensor2: NumPy array representing the second tensor
    - entropy1: Entropy value of tensor1
    - entropy2: Entropy value of tensor2
    - bins: Number of bins for the histogram
    - save_path: Path to save the histogram image

    Returns:
    - None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].hist(tensor1.flatten(), bins=bins)
    axes[0].set_title(f'Histogram of Downstream Segmentation feature maps\nEntropy: {entropy1:.2f}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(tensor2.flatten(), bins=bins)
    axes[1].set_title(f'Histogram of Multihead Autoencoder feature maps\nEntropy: {entropy2:.2f}')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


entropy_values_1 = []
entropy_values_2 = []

for i in range(5):
    feature_map_1 = np.load(f"{DIR1}/feature_maps_{i + 1}.npy")
    feature_map_2 = np.load(f"{DIR2}/feature_maps_{i + 1}.npy")

    # Normalize the feature maps
    # feature_map_1 = (feature_map_1 - np.min(feature_map_1)) / (np.max(feature_map_1) - np.min(feature_map_1))
    # feature_map_2 = (feature_map_2 - np.min(feature_map_2)) / (np.max(feature_map_2) - np.min(feature_map_2))

    entropy_1 = calculate_entropy(feature_map_1.flatten())
    entropy_2 = calculate_entropy(feature_map_2.flatten())
    # entropy1 = []
    # entropy2 = []
    # feature_map_1 = feature_map_1.squeeze()
    # feature_map_2 = feature_map_2.squeeze()
    # for j in range(feature_map_1.shape[0]):
    #     entropy1.append(entropy(feature_map_1[j].flatten()))
    #     entropy2.append(entropy(feature_map_2[j].flatten()))
    # entropy_1 = np.mean(entropy1)
    # entropy_2 = np.mean(entropy2)
    # Save histogram images with entropy values inside
    histogram_path = os.path.join(OUTPUT_DIR, f"histogram_comparison_{i + 1}.png")

    tensor_histogram(feature_map_1, feature_map_2, entropy_1, entropy_2, bins=100, save_path=histogram_path)

    # Save entropy values as text
    entropy_path_1 = os.path.join(OUTPUT_DIR, f"entropy_feature_map_1_{i + 1}.txt")
    entropy_path_2 = os.path.join(OUTPUT_DIR, f"entropy_feature_map_2_{i + 1}.txt")

    with open(entropy_path_1, 'w') as f:
        f.write(f"Entropy of feature map 1_{i + 1}: {entropy_1}\n")

    with open(entropy_path_2, 'w') as f:
        f.write(f"Entropy of feature map 2_{i + 1}: {entropy_2}\n")

print(entropy_values_1)
print(entropy_values_2)
