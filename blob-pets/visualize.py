import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from collections import defaultdict, Counter

from load_blob_pets import *

master = []
for batch_idx, data in enumerate(testloader):
    print(f"\nBatch {batch_idx}:")
    print()
    images, labels = map_labels(data)
    print(labels)
    
  
    first_image = images[0].permute(1, 2, 0).numpy()
    
    if first_image.max() <= 1.0:
        first_image = (first_image * 255).astype(np.uint8)
    
    small = []
    
    print("RGB values:")
    for row in range(4):
        for col in range(4):
            r, g, b = first_image[row, col]
            print(f"({r:3d},{g:3d},{b:3d})", end=" ")
            small.append([r, g, b])
        print()
    
    label_key = str(labels) if not isinstance(labels, (str, int)) else labels
    
    
    master.append({label_key: small})



#idea #1 average all values into 1 tuple of rgb
print(master[-1])

avg_rgb = []

for image in master:
    for key, rgb_list in image.items():
        # Convert to numpy array and calculate mean
        rgb_array = np.array(rgb_list)
        avg_values = rgb_array.mean(axis=0).tolist()
        rounded_values = [(int(val) // 10) * 10 for val in avg_values]
        avg_rgb.append({key: rounded_values})
    
print(avg_rgb[-1])

rgb_class_counts = defaultdict(list)

for item in avg_rgb:
    for class_label, rgb_values in item.items():
        rgb_tuple = tuple(rgb_values)  
        rgb_class_counts[rgb_tuple].append(class_label)

rgb_proportions = {}
unique_classes = set()

for rgb_tuple, class_list in rgb_class_counts.items():
    class_counts = Counter(class_list)
    total_count = len(class_list)

    unique_classes.update(class_counts.keys())
    
    proportions = {cls: count/total_count for cls, count in class_counts.items()}
    rgb_proportions[rgb_tuple] = proportions

unique_classes = sorted(list(unique_classes))
print(f"Classes found: {unique_classes}")

def rgb_to_ternary_coords(proportions, classes):
    if len(classes) != 3:
        print(f"Warning: Found {len(classes)} classes, ternary plot needs exactly 3")
        return None, None
    
    # Get proportions for each class (0 if not present)
    p1 = proportions.get(classes[0], 0)
    p2 = proportions.get(classes[1], 0)
    p3 = proportions.get(classes[2], 0)
    
    # Normalize to ensure they sum to 1
    total = p1 + p2 + p3
    if total == 0:
        return None, None
    
    p1, p2, p3 = p1/total, p2/total, p3/total
    
    # Convert to Cartesian coordinates for plotting
    x = 0.5 * (2*p2 + p3)
    y = (np.sqrt(3)/2) * p3
    
    return x, y


# Step 4: Create the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Draw the triangle
triangle_x = [0, 1, 0.5, 0]
triangle_y = [0, 0, np.sqrt(3)/2, 0]
ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)


# Only proceed if we have exactly 3 classes
if len(unique_classes) == 3:
    # Plot each RGB value
    x_coords = []
    y_coords = []
    
    for rgb_tuple, proportions in rgb_proportions.items():
        x, y = rgb_to_ternary_coords(proportions, unique_classes)
        if x is not None and y is not None:
            x_coords.append(x)
            y_coords.append(y)
    
    # Create scatter plot with uniform size and color
    scatter = ax.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.7, edgecolors='black')
    
    # Add class name labels at triangle corners
    ax.text(-0.05, -0.05, 'dog', fontsize=14, ha='right', va='top', 
            fontweight='bold')
    ax.text(1.05, -0.05, 'cat', fontsize=14, ha='left', va='top', 
            fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'bird', fontsize=14, ha='center', va='bottom', 
            fontweight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Class Distribution Simplex', fontsize=14)


plt.tight_layout()
plt.show()
