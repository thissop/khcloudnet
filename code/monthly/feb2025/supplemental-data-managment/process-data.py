from tkinter import image_names


def incorporate_thresholded_masks(threshold_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/threshold-with-26-and-others',
                                  batch_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/annotated-batches',
                                  key_file:str='code/monthly/feb2025/supplemental-data-managment/Supplemental Data Selection Management.csv',
                                  background_image_batch_dir:str="/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches",
                                  output_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/additional-data',
                                  output_mosaic_shape:list=[1536,1536], num_cutouts:int=9):
    '''
    
    modified (read: simplified and customized) version of code/monthly/jan2025/prepping-training-data/main.py for supplemental data

    '''
    
    import tarfile
    import os
    import pandas as pd 
    from tqdm import tqdm 
    import shutil 
    import numpy as np
    from PIL import Image
    import zipfile
    import cv2
    
    # Quality Control for Missing/Duplicate Images

    key_df=pd.read_csv(key_file)
    
    batch_arr = key_df['Batch'].to_numpy()
    name_arr = key_df['Name'].to_numpy()
    annotation_type_arr = key_df['Annotation Type'].to_numpy()

    # Q.C. I: Check for Duplicate Entries in Key 
    unique_elements, counts = np.unique(name_arr, return_counts=True)
    duplicates = unique_elements[counts > 1]

    if len(duplicates)!=0: 
        print(duplicates)
        raise Exception('Exception: Duplicate Entries in Key')

    # Q.C. II: Check if Any Term Appears in Filenames of Current Finalized Dataset 
    file_list = [entry.name for entry in os.scandir('/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/images')]# add this 
    term_found = any(any(term in filename for term in name_arr) for filename in file_list)

    if term_found:
        raise Exception('Exception: Filename Already Found Found in Data Directory') 

    # Q.C. III: Make Sure all Threshold Images have Threshold Mask and All Supervisely Images have Supervisely Mask

    threshold_image_names = name_arr[np.argwhere(annotation_type_arr=="Threshold")].flatten()
    supervisely_image_names = name_arr[np.argwhere(annotation_type_arr=="Supervisely")].flatten()

    for threshold_image_name in threshold_image_names:
        if not os.path.exists(os.path.join(threshold_dir, f'{threshold_image_name}-cloud-mask.png')): 
            raise Exception(f"Exception: {threshold_image_name} is missing a threshold mask!")
    
    for supervisely_image_name in supervisely_image_names: 
        found = False
        for khbatch_dir_number in list(set(batch_arr)):
            khbatch_dir = os.path.join(batch_dir, f'khbatch{khbatch_dir_number}')
            if os.path.exists(os.path.join(khbatch_dir, f'{supervisely_image_name}.png')): 
                found = True 
        
        if not found: 
            raise Exception(f'Exception: {supervisely_image_name} is missing supervisely annotation!')
        
    # Q.C. IV: Make sure size is correct 
    num_rows = int(num_cutouts**0.5)
    if num_cutouts%num_rows != 0: 
        raise Exception('sqrt(num_cutouts) needs to be an integer for square cutouts!')

    print('passed all tests')

    def make_in_correct_color_range():
        for khbatch_dir_number in list(set(batch_arr)):
            khbatch_dir = os.path.join(batch_dir, f'khbatch{khbatch_dir_number}')
            for mask_image in os.listdir(khbatch_dir):
                if 'png' in mask_image:
                    mask_image_path = os.path.join(khbatch_dir, mask_image)
                    img_arr = np.array(Image.open(mask_image_path).convert('L'))

                    mask = img_arr<0.5
                    img_arr[mask] = 0
                    img_arr[~mask] = 255

                    Image.fromarray(img_arr.astype(np.uint8)).save(mask_image_path, format="PNG", compress_level=0)

    #make_in_correct_color_range()
    #print('just fixed color ranges for supervisely')

    mask_cutout_dir = os.path.join(output_dir, 'masks')
    background_cutout_dir = os.path.join(output_dir, 'images')

    for image_name, mask_type, batch_number in tqdm(zip(name_arr, annotation_type_arr, batch_arr)):
            
            khbatch_dir = os.path.join(batch_dir, f'khbatch{batch_number}')

            if mask_type=='Threshold': 
                mask_image_path = os.path.join(threshold_dir, f'{image_name}-cloud-mask.png')
                
            else: 
                mask_image_path = os.path.join(khbatch_dir, f'{image_name}.png')

            background_image_path = os.path.join('/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/cutouts', f'{image_name}.png')

            mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
            background_image = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)

            # modify this so it only runs when the images don't exist 
            try: 
                resized_mask_image = cv2.resize(mask_image, output_mosaic_shape, interpolation=cv2.INTER_NEAREST)
                resized_background_image = cv2.resize(background_image, output_mosaic_shape, interpolation=cv2.INTER_AREA)
            except: 
                raise Exception(f'Exception: could not resize {image_name}!')

            # Generate Cutouts --> This needs to get moved to a standard raster_utils.py production routine. 

            cutout_size = int(output_mosaic_shape[0]/num_rows)
                
            row_index = 0
            for i in range(0, output_mosaic_shape[0], cutout_size): # this might/should be modified for non uniform shapes (rectangles, only works for squares for now)
                column_index = 0

                for j in range(0, output_mosaic_shape[0], cutout_size):
                    
                    file_name = f'{image_name}-{row_index}_{column_index}.png' 
                    mask_cutout_image_path = os.path.join(mask_cutout_dir, file_name)
                    background_cutout_image_path = os.path.join(background_cutout_dir, file_name)

                    if not os.path.exists(mask_cutout_image_path) and not os.path.exists(background_cutout_image_path): 

                        mask_cutout = resized_mask_image[i:i + cutout_size, j:j + cutout_size]
                        background_cutout = resized_background_image[i:i + cutout_size, j:j + cutout_size]

                        mask_cutout_image = Image.fromarray(mask_cutout)
                        background_cutout_image = Image.fromarray(background_cutout)

                        file_name = f'{image_name}-{row_index}_{column_index}.png' 

                        mask_cutout_image.save(mask_cutout_image_path, compress_level=0)
                        background_cutout_image.save(background_cutout_image_path, compress_level=0)

                    column_index += 1 

                row_index += 1 

#incorporate_thresholded_masks()

def combine_datasets(root_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready'):
    r"""only works first run"""
    
    import os 
    from tqdm import tqdm 
    import shutil 
    import numpy as np

    train_images_dir = os.path.join(root_dir, 'train_images')
    test_images_dir = os.path.join(root_dir, 'test_images')
    train_masks_dir = os.path.join(root_dir, 'train_masks')
    test_masks_dir = os.path.join(root_dir, 'test_masks')
    images_dir = os.path.join(root_dir, 'images')
    masks_dir = os.path.join(root_dir, 'masks')

    all_file_names = os.listdir(images_dir)
    image_names_set = np.array(list(set(["-".join(file_name.split('-')[:2]) for file_name in all_file_names if 'png' in file_name])))

    test_image_names = image_names_set[0:59]
    train_image_names = image_names_set[59:]
    
    for image_name in tqdm(train_image_names):
        for file_name in os.listdir(images_dir):
            if image_name in file_name: 
                shutil.copy(os.path.join(images_dir, file_name), os.path.join(train_images_dir, file_name))
                shutil.copy(os.path.join(masks_dir, file_name), os.path.join(train_masks_dir, file_name))

    for image_name in tqdm(test_image_names):
        for file_name in os.listdir(images_dir):
            if image_name in file_name: 
                shutil.copy(os.path.join(images_dir, file_name), os.path.join(test_images_dir, file_name))
                shutil.copy(os.path.join(masks_dir, file_name), os.path.join(test_masks_dir, file_name))



    #for test_image_name in tqdm(test_image_names):


combine_datasets()