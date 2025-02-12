
def combine_masks(path_one, path_two):
    import numpy as np 
    import os 
    from PIL import Image

    if os.path.exists(path_one) and os.path.exists(path_two): 
        threshold_image_arr = np.array(Image.open(path_two).convert('L'))
        batch_image_arr = np.array(Image.open(path_one).convert('L'))

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

path_one = '/Users/tkiker/Downloads/D3C1203-200258A009-0_2-cloud-mask.png'
path_two = '/Users/tkiker/Downloads/D3C1203-200258A009-0_2-cloud-mask-2.png'
combine_masks(path_one, path_two)