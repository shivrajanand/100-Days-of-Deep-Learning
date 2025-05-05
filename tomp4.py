import os
import cv2 
def numerical_sort(value):
    # Extract the numeric part of the filename
    return int(value.split('_')[1].split('.')[0])

def images_to_video(image_folder, output_video_file, frame_rate=15):
    # Get all image files from the folder and sort them numerically
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=numerical_sort)  # Sort filenames numerically based on the number in the name
    
    # Check if there are any images in the folder
    if not images:
        print("No images found in the folder.")
        return
    
    # Get the width and height from the first image to create the video
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    size = (width, height)

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
    out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, size)

    # Loop through each image and write it to the video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        out.write(img)  # Write the image to the video

        print(f"Adding frame {image_name} to video...")

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {output_video_file}")

# Usage Example:
image_folder = './tmp'  # Folder where your saved images are located
output_video_file = 'perceptron_animation.mp4'  # Output video file name
frame_rate = 10  # Lower the frame rate to slow down the video

# Convert images to video
images_to_video(image_folder, output_video_file, frame_rate)
