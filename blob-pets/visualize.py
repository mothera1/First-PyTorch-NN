import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict, Counter
import random

from load_blob_pets import *
from test_blob import *
from test_hinge import *

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
        rounded_values = [round(int(val) / 20) * 20 for val in avg_values]
        avg_rgb.append({key: rounded_values})
    
print(avg_rgb[-2])

print(predictions[-2])

pred_wrgb = []
for i in range(len(avg_rgb)):
    pred_wrgb.append({str(predictions[i]): list(avg_rgb[i].values())})

pred_wrgb = [{key: value[0] for key, value in item.items()} for item in pred_wrgb]

pred_wrgb_h = []
for i in range(len(avg_rgb)):
    pred_wrgb_h.append({str(predictions_h[i]): list(avg_rgb[i].values())})

pred_wrgb_h = [{key: value[0] for key, value in item.items()} for item in pred_wrgb_h]

def plot(avg_rgb):
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

        ids = [str(i) for i in range(1213)]

        
    return x_coords, y_coords
 
x_avg, y_avg = plot(avg_rgb)
x_pred, y_pred = plot(pred_wrgb)
x_h, y_h = plot(pred_wrgb_h)

print(len(x_avg))
print(len(x_pred))

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Draw the triangle
triangle_x = [0, 1, 0.5, 0]
triangle_y = [0, 0, np.sqrt(3)/2, 0]
ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)


# Create scatter plot with uniform size and color
random.seed(42)
indices = random.sample(range(len(x_avg)), 10)


x_avg = [x_avg[i] for i in indices]
y_avg = [y_avg[i] for i in indices]
x_pred = [x_pred[i] for i in indices]
y_pred = [y_pred[i] for i in indices]
x_h = [x_h[i] for i in indices]
y_h = [y_h[i] for i in indices]

ax.scatter(x_avg, y_avg, marker='o', c='blue', s=50, alpha=0.7, edgecolors='black')
ax.scatter(x_h, y_h, marker='v', c='green', s=50, alpha=0.7, edgecolors='black')
ax.scatter(x_pred, y_pred, marker='+', c='red', s=50, alpha=0.7, edgecolors='black')

ax.plot([0.5, 0.5, 0.75, 0.5, 0.25], [0, np.sqrt(3)/6, 0.433, np.sqrt(3)/6, 0.433], 'k--', lw=1)


for xa, ya, xp, yp in zip(x_avg, y_avg, x_pred, y_pred):
    ax.plot([xa, xp], [ya, yp], color='gray', linestyle='--', linewidth=1)

for xa, ya, xp, yp in zip(x_avg, y_avg, x_h, y_h):
    ax.plot([xa, xp], [ya, yp], color='gray', linestyle='--', linewidth=1)


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