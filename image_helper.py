import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image

def crop_big_image(big_image, box_idx=0):
    if box_idx == 0:
        return big_image.crop((448 + 256, 824, 960, 1080))
    if box_idx == 1:
        return big_image.crop((432 + 256, 808, 944, 1064))
    if box_idx == 2:
        return big_image.crop((440 + 256, 816, 952, 1072))
    if box_idx == 3:
        return big_image.crop((424 + 256, 800, 936, 1056))


def grid_crop(big_image, crop_size=32):
    small_images = []
    image_size = big_image.size

    for i in range(image_size[0] // crop_size):
        for j in range(image_size[1] // crop_size):
            box = (i * crop_size, j * crop_size, (i + 1) * crop_size, (j + 1) * crop_size)
            small_image = big_image.crop(box)
            small_images.append(small_image)

    return small_images

def display_grid_crop(grid_images, grid=(8, 8), image_size=32):
    background = Image.new('RGB',(grid[0] * image_size + grid[0], grid[1] * image_size + grid[1]), (255, 255, 255))
    for i in range(grid[0]):
        for j in range(grid[1]):
            offset = (i * image_size + i, j * image_size + j)
            background.paste(grid_images[i * grid[1] + j], offset)
    
    return background

def get_tensor(image, transform):
    image1 = crop_big_image(image, 0)
    image2 = crop_big_image(image, 1)
    image3 = crop_big_image(image, 2)
    image4 = crop_big_image(image, 3)

    grid_crop_trans1 = [transform(x) for x in grid_crop(image1)]
    grid_crop_trans2 = [transform(x) for x in grid_crop(image2)]
    grid_crop_trans3 = [transform(x) for x in grid_crop(image3)]
    grid_crop_trans4 = [transform(x) for x in grid_crop(image4)]

    image_tensor = torch.cat([torch.stack(grid_crop_trans1), torch.stack(grid_crop_trans2), torch.stack(grid_crop_trans3), torch.stack(grid_crop_trans4)])

    return image_tensor