import os
import shutil
import random

# Source directories (update if needed)
train_images_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/train_images'
train_masks_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/train_masks'
test_images_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/test_images'
test_masks_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/test_masks'

# Target directories
train_output_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/khcloudnet_train_10'
test_output_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/khcloudnet_test_10'

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Sampling function
def sample_and_copy(source_images_dir, source_masks_dir, output_dir, sample_fraction=0.1):
    images = [f for f in os.listdir(source_images_dir) if f.endswith('.png')]
    sample_size = max(1, int(len(images) * sample_fraction))
    sampled_images = random.sample(images, sample_size)

    print(f'Sampling {sample_size} images from {source_images_dir} to {output_dir}.')

    for img in sampled_images:
        src_img_path = os.path.join(source_images_dir, img)
        dst_img_path = os.path.join(output_dir, img)

        shutil.copy2(src_img_path, dst_img_path)

        # Find corresponding mask
        base_name = os.path.splitext(img)[0]
        mask_name = f'{base_name}.png'  # Assuming masks have the same base names in the source
        src_mask_path = os.path.join(source_masks_dir, mask_name)

        if os.path.exists(src_mask_path):
            dst_mask_name = f'{base_name}_annotation_and_boundary.png'
            dst_mask_path = os.path.join(output_dir, dst_mask_name)
            shutil.copy2(src_mask_path, dst_mask_path)
        else:
            print(f'Warning: Mask not found for {img}')

# Run sampling for both train and test
sample_and_copy(train_images_dir, train_masks_dir, train_output_dir, sample_fraction=0.1)
sample_and_copy(test_images_dir, test_masks_dir, test_output_dir, sample_fraction=0.1)

print('Sampling and copying completed.')
