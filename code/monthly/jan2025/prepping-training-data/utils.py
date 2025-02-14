
def combine_masks(path_one, path_two):
    """Path one is for supervisely image, path two is for threshold"""
    import numpy as np 
    import os 
    from PIL import Image

    if os.path.exists(path_one) and os.path.exists(path_two): 
        threshold_image_arr = np.array(Image.open(path_one).convert('L'))
        batch_image_arr = np.array(Image.open(path_two).convert('L'))

        mask = batch_image_arr<0.5
        batch_image_arr[mask] = 0
        batch_image_arr[~mask] = 255

        mask = threshold_image_arr > 100
        threshold_image_arr[mask] = 255
        threshold_image_arr[~mask] = 0

        merged = np.where(np.logical_or(batch_image_arr == 255, threshold_image_arr == 255), 255, 0)

        #os.remove(path_two)
        os.remove(path_two)

        # Save the image as PNG without compression
        Image.fromarray(merged.astype(np.uint8)).save(path_one, format="PNG", compress_level=0)

name = 'D3C1219-200603F002-0_3'
path_one = f'/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/threshold-with-26-and-others/{name}-cloud-mask.png'
path_two = f'/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/annotated-batches/khbatch26/dataset 2025-01-09 17-54-58/masks_machine/{name}.png'
combine_masks(path_one, path_two)