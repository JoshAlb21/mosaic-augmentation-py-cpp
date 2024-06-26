import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pickle

from src_python.utils import transforms


def make_bbox_valid(bbox, width, height, img_width, img_height) -> list:
    """
    Ensure that the bounding box is valid:
    - Bounding box coordinates are within image boundaries
    - Bounding box width and height are positive
    """
    # Ensure xmin <= xmax and ymin <= ymax
    xmin, ymin, xmax, ymax = bbox

    # If width or height is 0 or negative, adjust it
    if width <= 0:
        if xmax + 1 < img_width:
            xmax += 1
        elif xmin - 1 >= 0:
            xmin -= 1

    if height <= 0:
        if ymax + 1 < img_height:
            ymax += 1
        elif ymin - 1 >= 0:
            ymin -= 1

    # Ensure the bounding box is within image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)

    return [xmin, ymin, xmax, ymax]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, split_set:str, transforms=None, des_image_size=(512, 512), transform_settings:dict=None):

        if split_set not in ["train", "val"]:
            raise ValueError("split_set must be either 'train' or 'val'")
        self.root = root
        self.des_image_size = des_image_size
        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_path = os.path.join(root, "images", split_set)
        self.mask_path = os.path.join(root, "labels", split_set)
        self.imgs = list(sorted(os.listdir(self.img_path)))
        self.masks = list(sorted(os.listdir(self.mask_path)))

        self.manual_retrieve = False

        self.transforms = transforms(self, **transform_settings)
        # to manually retrieve an image without augmentation, required for moasic augmentation
        transform_settings["train"] = False
        self.manual_retrieve_transforms = transforms(self, **transform_settings)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_path, self.imgs[idx])
        mask_path = os.path.join(self.mask_path, self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        # with 0 being background
        mask = Image.open(mask_path)

        # Resize the image and mask
        img = img.resize(self.des_image_size, Image.BILINEAR) 
        mask = mask.resize(self.des_image_size, Image.NEAREST)

        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        if self.transforms is not None and not self.manual_retrieve:
            img, masks = self.transforms(img, masks)
        elif self.transforms is not None and self.manual_retrieve: # for mosaic augmentation
            img, masks = self.manual_retrieve_transforms(img, masks)

        # Retrieve obj_ids again, because they might have changed due to augmentation
        obj_ids = np.arange(1, masks.shape[0]+1) # -> [1, 2, 3, ...]

        # Get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        to_delete = []
        for i in range(num_objs):
            # Handle empty mask
            if masks[i].sum() == 0:
                #raise ValueError("Empty mask")
                # could be due to augmentation, therefore leave that mask empty
                #masks = np.delete(masks, i, axis=0)
                # TODO check if this is correct, better: delete mask and label for that particular layer
                #boxes.append([0, 0, 0, 0])
                to_delete.append(i)
                # we cannot delete the mask here, because we need to keep the same number of masks for the loop logic
                continue

            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # check for 0.0 coordinates
            for coor in [xmin, xmax, ymin, ymax]:
                if coor == 0.0:
                    coor = 0.001

            # check width and height
            width = xmax - xmin
            height = ymax - ymin
            if width < 1 or height < 1:
                # save img and mask for debugging
                #img.save(f"invalid_mask_{i}.png")
                # save ndarray as png
                #invalid_mask = Image.fromarray(masks[i].astype(np.uint8)*255)
                #invalid_mask.save(f"invalid_mask_{i}_mask.png")
                #raise ValueError(f"Width or height is smaller than 1. Index {i}, width {width}, height {height}")
            
                # Ensure that the bounding box is valid
                print("Handle invalid bbox.")
                xmin, ymin, xmax, ymax = make_bbox_valid([xmin, ymin, xmax, ymax], width, height, img.shape[2], img.shape[1])

            boxes.append([xmin, ymin, xmax, ymax])
        
        # We reverse the order of indices to ensure that deletions from the end
        # do not shift the positions of items yet to be deleted. This prevents
        # inadvertently skipping or incorrectly deleting items.
        for delete_i in reversed(to_delete):
            masks = np.delete(masks, delete_i, axis=0)
            obj_ids = np.delete(obj_ids, delete_i)
            num_objs -= 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(obj_ids, dtype=torch.int64)  # Use obj_ids as labels
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        #image_id = torch.tensor([idx])
        image_id = int(idx)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Important to remove unnecessary 1-dimension
        #target = squeeze_masks([target])[0]

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(data_set,
                  train:bool,
                  enable:bool,
                  mosaic:float=0.0, ):

    transform = []
    transform.append(transforms.PILToTensor())
    transform.append(transforms.ConvertImageDtype(torch.float))

    augm = []
    if train and not data_set.manual_retrieve and enable:
        augm.append(mosaic_augmentation.CustomMosaicMask(data_set, imgsz=640, p=mosaic, n=4) if mosaic else None)

        augm = [a for a in augm if a is not None]
        transform.extend(augm)

    return transforms.Compose(transform)
