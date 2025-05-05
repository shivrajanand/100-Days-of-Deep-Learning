import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from PIL import Image
import os
import tempfile
import shutil

# Simulated Data
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)

def step(x):
    return 1 if x >= 0 else 0

def perceptron(X, y):
    m = []
    b = []
    
    X = np.insert(X, 0, 1, axis=1)  # Add bias term (1) to the input data
    weights = np.ones(X.shape[1])
    lr = 0.1  # Learning rate
    epochs = 200
    for i in range(epochs):
        j = np.random.randint(0, 100)
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr * (y[j] - y_hat) * X[j]
        
        m.append(-(weights[1] / weights[2]))  # Slope
        b.append(-(weights[0] / weights[2]))  # Intercept
    return m, b

m, b = perceptron(X, y)

# Subplot setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Decision boundary
x_i = np.linspace(-3, 3, 100)  # Define the range of x for the decision boundary plot
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100, edgecolors='k')
line, = ax1.plot(x_i, m[0] * x_i + b[0], 'r', linewidth=2)  # Initial decision boundary line
title = ax1.set_title("Epoch 1")

# Set axis limits for the decision boundary plot
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)

# Subplot 2: m and b growth over time
ax2.set_title("Growth of m and b")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Value')
ax2.set_xlim(0, len(m))  # Set x-axis limits to cover the range of epochs
ax2.set_ylim(min(np.min(m), np.min(b)) - 0.1, max(np.max(m), np.max(b)) + 0.1)  # Dynamic y-axis limits
line_m, = ax2.plot([], [], 'b', label="m (Slope)", linewidth=2)
line_b, = ax2.plot([], [], 'g', label="b (Intercept)", linewidth=2)
ax2.legend()

# Create a temporary directory to store frames
temp_dir = "./tmp"

# Create a list to store the filenames of saved images for later display
saved_frames = []

# Update function for animation
def update(i):
    line.set_ydata(m[i] * x_i + b[i])
    title.set_text(f"Epoch {i + 1}")
    line_m.set_data(np.arange(i+1), m[:i+1])
    line_b.set_data(np.arange(i+1), b[:i+1])

    # Save each frame to a temporary file in the temp_dir
    plt.draw()
    tmp_filename = os.path.join(temp_dir, f"frame_{i}.png")
    plt.savefig(tmp_filename, bbox_inches='tight')
    
    # Append the filename to saved_frames
    saved_frames.append(tmp_filename)

    return line, title, line_m, line_b

# Create the animation by updating for each epoch
for i in range(len(m)):
    update(i)

# Now that all frames are saved, load them into a list for the GIF creation or display
frames = []
for filename in saved_frames:
    try:
        frames.append(Image.open(filename))
    except Exception as e:
        print(f"Error opening frame {filename}: {e}")

# Create the animation using the saved frames
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))

def show_frame(i):
    ax2.clear()  # Clear the previous frame
    ax2.imshow(frames[i])
    ax2.axis('off')  # Turn off axis for cleaner look
    ax2.set_title(f"Epoch {i + 1}")

# Display the animation
ani = FuncAnimation(fig2, show_frame, frames=len(frames), interval=100, repeat=False)
plt.show()

