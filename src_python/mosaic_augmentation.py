import torch
import numpy as np
from torchvision.transforms import functional as F
import random
import pickle

from src_python.visualize import plot_image_masks_boxes

def ensure_three_channels(mask):
    """Ensure that the mask has three channels, filling missing channels with zeros."""
    c, h, w = mask.shape
    if c == 3:
        return mask
    elif c == 2:
        return torch.cat([mask, torch.zeros(1, h, w)], dim=0)
    elif c == 1:
        return torch.cat([mask, torch.zeros(2, h, w)], dim=0)
    else:
        raise ValueError(f"Unexpected number of channels: {c}")


class CustomMosaic(torch.nn.Module):
    '''Mosaic augmentation.

    Causion: Only implemented for mask targets. NOT IMPLEMENTED FOR BBOX TARGETS.
    '''

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        super().__init__()
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        assert n in (4, 9), 'grid must be equal to 4 or 9.'
        self.dataset = dataset
        self.imgsz = imgsz
        self.p = p
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n

    def forward(self, image, target):
        # If the random number is greater than self.p, return the original image and target.
        if torch.rand(1) > self.p:
            return image, target
        
        # Fetch other images and their targets here to create the mosaic.
        other_images, other_targets = self._fetch_other_samples()

        if self.n == 4:
            return self._mosaic4(image, target, other_images, other_targets)
        else:
            return self._mosaic9(image, target, other_images, other_targets)

    def _fetch_other_samples(self):
        """Fetch random samples from the dataset."""

        self.dataset.manual_retrieve = True # otherwise augmentation will be applied to each subimage
        other_images = []
        other_targets = []
        for _ in range(self.n - 1):
            idx = torch.randint(0, len(self.dataset), (1,)).item()
            image, target = self.dataset[idx]
            other_images.append(image)
            other_targets.append(target)
        self.dataset.manual_retrieve = False

        return other_images, other_targets
    
    def place_image(self, img, mosaic_img, x1, y1, x2, y2, target, padw, padh):

        print(f"x1: {x1}, x2: {x2}, width of slice: {x2-x1}, width of img: {img.shape[1]}")
        print(f"y1: {y1}, y2: {y2}, height of slice: {y2-y1}, height of img: {img.shape[2]}")
        # Place the image on the mosaic canvas
        mosaic_img[:, y1:y2, x1:x2] = img

        # Update bounding boxes
        target["boxes"][:, 0::2] += padw
        target["boxes"][:, 1::2] += padh
        
        # Update masks
        h, w = img.shape[2], img.shape[1]
        new_masks = torch.zeros((target["masks"].shape[1], self.imgsz * 2, self.imgsz * 2), dtype=torch.uint8)
        new_masks[:, y1:y1+h, x1:x1+w] = target["masks"]
        target["masks"] = new_masks

        return target

    def _mosaic4(self, image, target, other_images, other_targets):
        # Create a canvas for mosaic
        #mosaic_img = torch.zeros(3, self.imgsz * 2, self.imgsz * 2, dtype=torch.float32)
        # Desired color values
        fill_color = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        mosaic_img = fill_color.repeat(1, self.imgsz * 2, self.imgsz * 2)

        yc, xc = [int(random.uniform(-x, 2 * self.imgsz + x)) for x in self.border]  # mosaic center x, y
        
        # Place each image on the mosaic
        all_targets = []
        
        # Handling the main image
        h, w = image.shape[1], image.shape[2]
        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        mosaic_img[:, y1a:y2a, x1a:x2a] = image[:, y1b:y2b, x1b:x2b]
        target = self._update_labels(target, x1a - x1b, y1a - y1b)
        all_targets.append(target)
        
        # Handling the other images
        for i, (img, tgt) in enumerate(zip(other_images, other_targets)):
            h, w = img.shape[1], img.shape[2]
            if i == 0:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.imgsz * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 1:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.imgsz * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 2:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.imgsz * 2), min(self.imgsz * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            mosaic_img[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]
            tgt = self._update_labels(tgt, x1a - x1b, y1a - y1b)
            all_targets.append(tgt)
        
        # Aggregate all targets
        combined_target = self.combine_targets(all_targets, mosaic_img.shape, target["image_id"])

        return mosaic_img, combined_target

    def _mosaic9(self, image, target):
        raise NotImplementedError
    
    def _update_labels(self, target, padw, padh):
        """Update labels based on mosaic padding."""
        # Update bounding boxes
        target["boxes"][:, 0::2] += padw
        target["boxes"][:, 1::2] += padh
        
        # Update masks
        h, w = target["masks"].shape[-2:]  # Get height and width of the masks
        new_masks = torch.zeros((target["masks"].shape[0], self.imgsz * 2, self.imgsz * 2), dtype=torch.uint8)
        
        # Calculate start and end indices for mosaic and mask
        mosaic_y1 = max(padh, 0)
        mosaic_y2 = min(padh + h, new_masks.shape[1])
        mosaic_x1 = max(padw, 0)
        mosaic_x2 = min(padw + w, new_masks.shape[2])

        mask_y1 = max(-padh, 0)
        mask_y2 = h + min(self.imgsz * 2 - (padh + h), 0)
        mask_x1 = max(-padw, 0)
        mask_x2 = w + min(self.imgsz * 2 - (padw + w), 0)
        
        new_masks[:, mosaic_y1:mosaic_y2, mosaic_x1:mosaic_x2] = target["masks"][:, mask_y1:mask_y2, mask_x1:mask_x2]
        target["masks"] = new_masks

        return target

    @staticmethod
    def combine_targets(targets, img_shape, img_id):
        # Initialize empty lists for each target attribute
        combined_boxes = []
        combined_labels = []
        combined_masks = []
        combined_iscrowd = []
        
        for target in targets:
            # Clip boxes
            boxes = target['boxes']
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_shape[2])  # clamp x coordinates
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_shape[1])  # clamp y coordinates
            combined_boxes.append(boxes)
            
            # Append other attributes
            combined_labels.append(target['labels'])
            combined_masks.append(target['masks'])
            combined_iscrowd.append(target['iscrowd'])
        
        # Concatenate lists to form tensors
        combined_boxes = torch.cat(combined_boxes, dim=0)
        combined_labels = torch.cat(combined_labels, dim=0)
        combined_masks = torch.cat(combined_masks, dim=0)
        combined_iscrowd = torch.cat(combined_iscrowd, dim=0)
        
        # Recalculate areas
        areas = (combined_boxes[:, 3] - combined_boxes[:, 1]) * (combined_boxes[:, 2] - combined_boxes[:, 0])
        
        # Create combined target dictionary
        combined_target = {
            'boxes': combined_boxes,
            'labels': combined_labels,
            'masks': combined_masks,
            'image_id': img_id,
            'area': areas,
            'iscrowd': combined_iscrowd
        }
        
        return combined_target


def validate_mosaic_output(original_image, original_target, mosaic_image, mosaic_target):
    """
    Validate if the mosaic_image and mosaic_target maintain the properties and structure of the original ones.
    
    Parameters:
    - original_image: Original image before mosaic augmentation.
    - original_target: Original target before mosaic augmentation.
    - mosaic_image: Image after mosaic augmentation.
    - mosaic_target: Target after mosaic augmentation.

    Returns:
    - validation_result: A dictionary containing validation results.
    """
    validation_result = {
        'image_dtype': False,
        'target_keys': False,
        'target_shapes': True  # default to True, will be set to False if any shape mismatch is found
    }

    # Validate image dtype
    validation_result['image_dtype'] = original_image.dtype == mosaic_image.dtype

    # Validate target keys
    validation_result['target_keys'] = set(original_target.keys()) == set(mosaic_target.keys())
    
    # Validate shapes of target components
    for key in original_target.keys():
        if isinstance(original_target[key], torch.Tensor) and isinstance(mosaic_target[key], torch.Tensor):
            if original_target[key].shape[1:] != mosaic_target[key].shape[1:]:
                validation_result['target_shapes'] = False
                break

    return validation_result


if __name__ == '__main__':

    from . import visualize
    from .dataset_w_mosaic import MyDataset, get_transform

    mock_root = "/Users/joshuaalbiez/Documents/python/mosaic-augmentation-py-cpp/data/TestDataset"
    transform_settings = {
        "train": True,
        "mosaic": 0.0,
        "enable" : True
    }
    dataset = MyDataset(mock_root, 'train', get_transform, transform_settings=transform_settings)

    for i in range(5):
        img, target = dataset[i]    
        img_m, target_m = CustomMosaic(dataset, imgsz=640, p=1.0, n=4)(img, target)

        print(img_m.min(), img_m.max())
        print(img_m.dtype)
        print(img_m.shape)

        print(target_m["masks"][0].shape)

        plot_image_masks_boxes(img_m, target_m)

    validation_res = validate_mosaic_output(img, target, img_m, target_m)
    print(validation_res)

    print(target['masks'])
    print(target_m['masks'])