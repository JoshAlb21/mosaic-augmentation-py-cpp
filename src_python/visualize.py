import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors

class RectangleTransform:
    def __init__(self, size=400, translate_factor=0.7, scale_factor=1.0):
        self.size = size
        self.translate_factor = translate_factor
        self.scale_factor = scale_factor
        
        # Original rectangle centered at (0,0)
        half_size = size / 2
        self.rect = np.array([[-half_size, -half_size], [half_size, -half_size], [half_size, half_size], [-half_size, half_size], [-half_size, -half_size]])
        
        # Center of Gravity (COG) of the rectangle is at (0,0)
        self.cog = [0, 0]
    
    def apply_transformation(self):
        # Calculate Maximum and Minimum possible translations
        self.x_max_translation = self.translate_factor * self.size
        self.x_min_translation = -self.translate_factor * self.size
        self.y_max_translation = self.translate_factor * self.size
        self.y_min_translation = -self.translate_factor * self.size
        
        # Apply all four combinations of translations and scaling
        self.rects_transformed = {
            "Max_X_Max_Y": (self.rect * self.scale_factor) + [self.x_max_translation, self.y_max_translation],
            "Max_X_Min_Y": (self.rect * self.scale_factor) + [self.x_max_translation, self.y_min_translation],
            "Min_X_Max_Y": (self.rect * self.scale_factor) + [self.x_min_translation, self.y_max_translation],
            "Min_X_Min_Y": (self.rect * self.scale_factor) + [self.x_min_translation, self.y_min_translation]
        }
        
        # COGs for all transformed rectangles
        self.cogs_transformed = {
            key: np.mean(val[:-1], axis=0) for key, val in self.rects_transformed.items()
        }
        
    def visualize(self):
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Original Rectangle
        ax.plot(self.rect[:, 0], self.rect[:, 1], label="Original Rectangle", color="blue")
        ax.scatter(*self.cog, color="blue", marker="o", label="COG Original")
        
        # Transformed Rectangles
        colors = ["green", "orange", "purple", "red"]
        for (label, rect), color in zip(self.rects_transformed.items(), colors):
            ax.plot(rect[:, 0], rect[:, 1], label=label, color=color)
            ax.scatter(*self.cogs_transformed[label], color=color, marker="o")
        
        # Coordinate Origin
        ax.scatter(0, 0, color="black", marker="x", label="Origin")
        
        ax.legend()
        ax.set_xlim(-700, 700)
        ax.set_ylim(-700, 700)
        ax.axvline(0, color='grey',linewidth=0.5)
        ax.axhline(0, color='grey',linewidth=0.5)
        ax.set_title(f"Visualization of Translations and Scaling (translate_factor = {self.translate_factor}, scale_factor = {self.scale_factor})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()

class MosaicVisualization:
    def __init__(self, images, imgsz=640, n=4):
        assert n in (4, 9), 'grid must be equal to 4 or 9.'
        self.images = images
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.crops = []

    def _mosaic4(self):
        """Create a 2x2 image mosaic."""
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        mosaic = np.zeros((s * 2, s * 2, 3), dtype=np.uint8)
        
        for i in range(4):
            img = self.images[i]
            h, w, _ = img.shape

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            self.crops.append(((x1b, y1b), (x2b, y2b)))
        return mosaic

    # This function can be expanded similarly for _mosaic9
    # For brevity, I'll stick to the 2x2 mosaic for now

    def visualize(self):
        mosaic_img = self._mosaic4()

        # Plotting the final mosaic image
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mosaic_img)
        ax.set_title("Mosaic Image")
        plt.show()

        # Plotting each image with the used crop highlighted
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, img in enumerate(self.images):
            axes[i].imshow(img)
            x1, y1 = self.crops[i][0]
            x2, y2 = self.crops[i][1]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].set_title(f"Image {i+1} with crop")
        plt.tight_layout()
        plt.show()


def plot_image_masks_boxes(img, target, object_id=None, save_path=None):
    """
    Display an image with overlaid masks and bounding boxes.

    Parameters:
    - img: The original image.
    - target: Dictionary containing masks and bounding boxes.
    - object_id (optional): ID of the object to be displayed. If not provided, all objects will be displayed.
    - save_path (optional): Path to save the plotted image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Normalize image for display
    img = img - img.min()
    img = img / img.max()

    ax.imshow(img.permute(1, 2, 0).numpy())

    # If object_id is specified, filter masks and boxes based on labels
    if object_id is not None:
        indices = (target["labels"] == object_id).nonzero(as_tuple=True)[0]
        masks_to_plot = target["masks"][indices]
        boxes_to_plot = target["boxes"][indices]
    else:
        masks_to_plot = target["masks"]
        boxes_to_plot = target["boxes"]

    # Generate unique colors using a colormap
    num_masks = len(masks_to_plot)
    colormap = plt.cm.get_cmap('hsv', num_masks)
    colors = [colormap(i) for i in range(num_masks)]

    # Set axis limits based on the original image size
    ax.set_xlim(0, img.shape[2])
    ax.set_ylim(img.shape[1], 0)

    # Iterate over masks and overlay them on the image
    for i, mask in enumerate(masks_to_plot):
        rgba_mask = np.zeros((*mask.shape, 4))
        rgba_mask[..., :3] = colors[i][:3]
        rgba_mask[..., 3] = mask.numpy()  # Assuming mask is a torch tensor
        ax.imshow(rgba_mask)

    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=400)
    else:
        plt.show()


if __name__ == "__main__":
    # Create an instance of the class with translate_factor set to 1.0 and scale_factor set to 0.5, then visualize the transformations
    rect_transform_scale = RectangleTransform(translate_factor=1.0, scale_factor=0.5)
    rect_transform_scale.apply_transformation()
    rect_transform_scale.visualize()

    # Sample images for visualization (Using colored patches for simplicity)
    sample_images = [np.full((400, 400, 3), color, dtype=np.uint8) for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]]

    # Create an instance of the visualization class and visualize the mosaic process
    mosaic_viz = MosaicVisualization(sample_images)
    mosaic_viz.visualize()