#NOTE(Jesse): The premise of this script is to train a CNN with the provided prepared training data.
# The inputs are assumed to have been produced using the stage_1 script

#NOTE(Jesse): The format of the input arguments are:
# argv[1]: Base directory of training data
# argv[2]: unix glob match match to select annotation tif images
# -- Args below are only used during monte carlo training.  Otherwise, model number is defaulted to 1, and training percentage is 100.
# argv[3]: The "model number" used with monte carlo training
# argv[4]: Training percentage to use for training used with monte carlo training
# argv[5 ...]: A list of mosaic bands used with monte carlo training

training_data_fp = "/path/to/base/path"
#training_data_glob_match = "/train_images_10/*_annotation_and_boundary.tif"
training_data_glob_match = "train_images_10/*_annotation_and_boundary.tif"
#training_data_glob_match = "/*/*_annotation_and_boundary_*.tif" #NOTE(Jesse): This string is appended to training_data_fp and is globbed to locate all matching files under the fp directory.

model_weights_fp = None #NOTE(Jesse): Set to a weights training file for post-train

unet_weights_fn_template = "unet_{}.weights.h5"

a100_partition = False

epochs = 15
steps_per_epoch = 1024

evaluation_count_target = 2 ** 12 #NOTE(Jesse): Number of evaluation steps between train epochs. 
global_batches_per_run = 64 #NOTE(Jesse): Upload N number of batches per fit().  Try halving if OOM.  Should be the last variable to adjust if required.

##
# Do Not Adjust global variables declared below this comment!
##
from os import environ
environ['OPENBLAS_NUM_THREADS'] = '2' #NOTE(Jesse): Insanely, _importing numpy_ will spawn a threadpool of num_cpu threads and this is the 'preferred' mechanism to limit the thread count.

from unet.config import unet_config
from predict_utilities import standardize_inplace

from shutil import copy
from os import remove

from numpy import dot, newaxis, zeros, float32, uint16, uint32, uint8, uint64, isnan, int8, ndarray, prod, concatenate, array_equal#,roll, sin, cos, pi, array
from numpy.random import default_rng
from gc import collect

global_local_tmp_dir = environ.get("LOCAL_TMPDIR")
if global_local_tmp_dir is None:
    global_local_tmp_dir = environ.get("TSE_TMPDIR")

model_training_percentage = 100
model_number = 1

#NOTE(Jesse): The global_ objects below are for multiplrocessing pool workers

global_shared_raster_batch_shape = None
global_shared_raster_shape = None

global_shared_anno_boun_batch_shape =  None
global_shared_anno_boun_shape = None

global_shared_raster_count = None

global_unet_context = None

global_training_frames_count = None
global_training_empty_frames_count = None
global_validation_frames_count = None
global_validation_empty_frames_count = None

global_batch_queue_depth = 2 #NOTE(Jesse): Do not touch

global_lr_multiplier = 1.0
global_momentum1_multiplier = 1.0
global_momentum2_multiplier = 1.0
global_wd_multiplier = 1.0
global_lr_schedule_toggle = False
global_opt_adamw_toggle = False

global_opt_params = None

import faulthandler
faulthandler.enable()

from osgeo import gdal
gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("GDAL_CACHEMAX", "0")

#NOTE(Jesse): For debug printing
#job_host = environ.get("HOST")
#if job_host is not None:
#    print(job_host)

user_id = environ.get("USER")
try:
    sm_suffix = "_" + user_id + "_" + environ["SLURM_JOB_ID"]
    sm_suffix += ("_" + environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in environ else ""
except Exception as e:
    sm_suffix = "_local"

if global_local_tmp_dir is not None:
    global_local_tmp_dir += '/' + sm_suffix

global_raster_sm_name = "raster" + sm_suffix
global_anno_boun_sm_name = "anno_boun" + sm_suffix

global_raster_batch_sm_name = "raster_batch" + sm_suffix
global_anno_boun_batch_sm_name = "anno_boun_batch" + sm_suffix

global_tf_indices_sm_name = "tf_batch_indices" + sm_suffix

#print(global_tf_indices_sm_name)
clear_shared_memory_string = f"find /dev/shm -user {user_id} | grep '{sm_suffix}' | xargs rm -f"

def init_pool_global_variables(in_raster_shape, in_anno_boun_shape, in_raster_batch_shape, in_anno_boun_batch_shape, in_shared_raster_count, in_unet_context, training_frames_count, training_empty_frames_count, validation_frames_count, validation_empty_frames_count):
    global global_shared_raster_shape, global_shared_anno_boun_shape, global_shared_raster_batch_shape, global_shared_anno_boun_batch_shape, global_shared_raster_count, global_unet_context, global_training_frames_count, global_training_empty_frames_count, global_validation_frames_count, global_validation_empty_frames_count
    global_shared_raster_shape, global_shared_anno_boun_shape, global_shared_raster_batch_shape, global_shared_anno_boun_batch_shape, global_shared_raster_count, global_unet_context, global_training_frames_count, global_training_empty_frames_count, global_validation_frames_count, global_validation_empty_frames_count = in_raster_shape, in_anno_boun_shape, in_raster_batch_shape, in_anno_boun_batch_shape, in_shared_raster_count, in_unet_context, training_frames_count, training_empty_frames_count, validation_frames_count, validation_empty_frames_count

    environ['OPENBLAS_NUM_THREADS'] = '2'
    from cv2 import INTER_NEAREST, INTER_AREA, warpAffine, getStructuringElement, getRotationMatrix2D, morphologyEx, MORPH_CLOSE, MORPH_OPEN, MORPH_RECT, MORPH_ELLIPSE#,INTER_LINEAR, INTER_MAX

def init_load_raster_global_variables(in_unet_context, in_global_shared_raster_shape, in_global_shared_anno_boun_shape):
    global global_unet_context
    global_unet_context = in_unet_context

    global global_shared_raster_shape, global_shared_anno_boun_shape
    global_shared_raster_shape, global_shared_anno_boun_shape = in_global_shared_raster_shape, in_global_shared_anno_boun_shape

def process_load_raster_and_anno_files(in_anno_boun_fp, idx):
    label_shape = global_unet_context.label_shape
    batch_input_shape = global_unet_context.batch_input_shape
    raster_shape = global_unet_context.raster_shape
    raster_band_indices = global_unet_context.raster_band_indices

    raster_patches_shm = shared_memory.SharedMemory(name=global_raster_sm_name)
    anno_boun_patches_shm = shared_memory.SharedMemory(name=global_anno_boun_sm_name)

    all_raster_patches = ndarray(global_shared_raster_shape, dtype=float32, buffer=raster_patches_shm.buf)
    all_anno_boun_patches = ndarray(global_shared_anno_boun_shape, dtype=uint8, buffer=anno_boun_patches_shm.buf)

    r_fp = in_anno_boun_fp.replace("_annotation_and_boundary", "")
    anno_boun_fp = in_anno_boun_fp
    if global_local_tmp_dir:
        #NOTE(Jesse): Use the directory and filename as a kind of namespace when copying to local tempdir
        # otherwise, other jobs on the same node might be loading / removing a file with the same name
        # from a different input source (same tile and same name but different date)
        tmp_in_anno_boun_fn = in_anno_boun_fp.replace('/', '_')
        tmp_r_fn = r_fp.replace('/', '_')

        anno_boun_fp = copy(in_anno_boun_fp, global_local_tmp_dir + tmp_in_anno_boun_fn)
        r_fp = copy(r_fp, global_local_tmp_dir + tmp_r_fn)

    r_ds = gdal.Open(r_fp)
    assert r_ds.RasterCount >= batch_input_shape[-1]

    for r_i in range(r_ds.RasterCount):
        assert r_ds.GetRasterBand(r_i + 1).DataType == gdal.GDT_UInt16#gdal.GDT_Byte

    in_raster_shape = (r_ds.RasterXSize, r_ds.RasterYSize)
    if not (in_raster_shape == (raster_shape[0], raster_shape[1])):
        print(f"[WARN]: Expected square shape of 512x512, got {r_ds.RasterXSize, r_ds.RasterYSize} from {r_fp}.  Skipping")
        assert False
    
    raster =    all_raster_patches[idx] #zeros((raster_shape[0], raster_shape[1], batch_input_shape[-1]), dtype=float32)
    anno_boun = all_anno_boun_patches[idx] #zeros((raster_shape[0], raster_shape[1], 1), dtype=uint8)
    for b_idx, b in enumerate(raster_band_indices):
        r_ds.GetRasterBand(b).ReadAsArray(buf_obj=raster[..., b_idx])

    standardize_inplace(raster)

    r_ds = None

    ab_ds = gdal.Open(anno_boun_fp)
    assert ab_ds.RasterCount == 1
    assert ab_ds.RasterYSize == ab_ds.RasterXSize == label_shape[0] == in_raster_shape[0]

    ab_ds.ReadAsArray(buf_obj=anno_boun[..., 0])
    ab_ds = None

    anno_boun[anno_boun > 1] = 10

    assert not isnan(anno_boun).any(), in_anno_boun_fp
    assert not isnan(raster).any(), r_fp

    is_empty = (anno_boun > 0).sum(dtype=uint64) == 0

    if global_local_tmp_dir:
        remove(anno_boun_fp)
        remove(r_fp)

    raster_patches_shm.close()
    anno_boun_patches_shm.close()

    return is_empty


def process_uniform_patch_generator(write_base_idx):
    batch_size = global_unet_context.batch_size
    batch_input_shape = global_unet_context.batch_input_shape
    batch_label_shape = global_unet_context.batch_label_shape
    raster_shape = global_unet_context.raster_shape

    raster_patches_shm = shared_memory.SharedMemory(name=global_raster_sm_name)
    anno_boun_patches_shm = shared_memory.SharedMemory(name=global_anno_boun_sm_name)

    batch_raster_shm = shared_memory.SharedMemory(name=global_raster_batch_sm_name)
    batch_anno_shm = shared_memory.SharedMemory(name=global_anno_boun_batch_sm_name)

    tf_batch_indices_sm = shared_memory.SharedMemory(name=global_tf_indices_sm_name)

    all_raster_patches = ndarray(global_shared_raster_shape, dtype=float32, buffer=raster_patches_shm.buf)
    all_anno_boun_patches = ndarray(global_shared_anno_boun_shape, dtype=uint8, buffer=anno_boun_patches_shm.buf)

    shared_raster_batch = ndarray(global_shared_raster_batch_shape, dtype=float32, buffer=batch_raster_shm.buf)
    shared_anno_boun_batch = ndarray(global_shared_anno_boun_batch_shape, dtype=uint8, buffer=batch_anno_shm.buf)

    shared_tf_indices_batch = ndarray(global_batch_queue_depth * global_batches_per_run * batch_size, dtype=uint32, buffer=tf_batch_indices_sm.buf)

    assert all_raster_patches.shape[0] == all_anno_boun_patches.shape[0]

    raster_patches = all_raster_patches[global_training_frames_count:][:global_validation_frames_count]
    anno_boun_patches = all_anno_boun_patches[global_training_frames_count:][:global_validation_frames_count]

    raster_empty_patches = all_raster_patches[global_training_frames_count + global_validation_frames_count + global_training_empty_frames_count:]

    rng = default_rng()
    randint = rng.integers

    raster_xy_size = raster_shape[0]
    batch_xy_size = batch_input_shape[0]

    rnd_count = 128

    patches_per_run = batch_size
    empty_patches_this_idx = randint(0, 10, rnd_count, dtype=uint8)
    patch_offsets = randint(0, raster_xy_size - batch_xy_size, (rnd_count, 2), dtype=uint16)

    patch_indices = randint(0, raster_patches.shape[0], rnd_count, dtype=uint16)

    if raster_empty_patches.shape[0] > 0:
        empty_patch_indices = randint(0, raster_empty_patches.shape[0], rnd_count, dtype=uint16)
    else:
        empty_patch_indices = None

    tf_indices = zeros(patches_per_run, dtype=uint32)

    raster_batch = zeros((patches_per_run, *batch_input_shape), dtype=float32)
    anno_boun_batch = zeros((patches_per_run, *batch_label_shape), dtype=uint8)

    empty_anno_bound = zeros(anno_boun_patches.shape[1:], dtype=uint8)

    maximum_empty_patch_regions_count_per_map = max(1, batch_size // 16)

    empty_patch_regions_count = 0
    batch_idx = 0
    rnd_idx = 0
    while batch_idx < patches_per_run:
        choice_raster_patches = raster_patches
        choice_patch_indices = patch_indices
        #choice_fps = raster_fps

        empty_patch_this_idx = False
        if empty_patch_indices is not None:
            empty_patch_this_idx = empty_patches_this_idx[rnd_idx] == 0 
            if empty_patch_this_idx:
                if empty_patch_regions_count >= maximum_empty_patch_regions_count_per_map:
                    empty_patch_this_idx = False
                else:
                    choice_raster_patches = raster_empty_patches
                    choice_patch_indices = empty_patch_indices
                    #choice_fps = raster_empty_fps

                    empty_patch_regions_count += 1

        skipped = 0
        while True:
            patch_idx = choice_patch_indices[rnd_idx]
            raster = choice_raster_patches[patch_idx]
            anno_boun = empty_anno_bound if empty_patch_this_idx else anno_boun_patches[patch_idx]
            
            (y, x) = patch_offsets[rnd_idx]

            anno_boun_patch = anno_boun[y:y + batch_xy_size, x:x + batch_xy_size]
            raster_patch = raster[y:y + batch_xy_size, x:x + batch_xy_size]

            rnd_idx += 1
            if rnd_idx == rnd_count:
                empty_patches_this_idx[:] = randint(0, 10, rnd_count, dtype=uint8)
                patch_offsets[:] = randint(0, raster_xy_size - batch_xy_size, (rnd_count, 2), dtype=uint16)

                patch_indices[:] = randint(0, raster_patches.shape[0], rnd_count, dtype=uint16)
                if empty_patch_indices is not None:
                    empty_patch_indices[:] = randint(0, raster_empty_patches.shape[0], rnd_count, dtype=uint16)
                rnd_idx = 0

            if not empty_patch_this_idx:
                if (anno_boun_patch == 1).sum(dtype=uint64) == 0:
                    if skipped > 6:
                        break

                    skipped += 1
                    continue

            break

        #assert not any(isnan(batch_pan_ndvi)), tf
        #assert not any(isnan(batch_anno_boun)), tf

        raster_batch[batch_idx] = raster_patch
        standardize_inplace(raster_batch[batch_idx])
        anno_boun_batch[batch_idx] = anno_boun_patch
        tf_indices[batch_idx] = patch_idx if not empty_patch_this_idx else global_training_frames_count + global_validation_frames_count + patch_idx
        batch_idx += 1

    #standardize_batch_inplace(raster_batch)
    shared_raster_batch[write_base_idx:write_base_idx+patches_per_run] = raster_batch
    shared_anno_boun_batch[write_base_idx:write_base_idx+patches_per_run] = anno_boun_batch
    shared_tf_indices_batch[write_base_idx:write_base_idx+patches_per_run] = tf_indices

    raster_patches_shm.close()
    anno_boun_patches_shm.close()
    batch_raster_shm.close()
    batch_anno_shm.close()
    tf_batch_indices_sm.close()


def process_uniform_with_data_aug_random_patch(write_base_idx):
    from cv2 import INTER_NEAREST, INTER_AREA, warpAffine, getStructuringElement, getRotationMatrix2D, morphologyEx, MORPH_OPEN, MORPH_CLOSE, MORPH_RECT, MORPH_ELLIPSE#,INTER_LINEAR, INTER_MAX

    batch_size = global_unet_context.batch_size
    batch_input_shape = global_unet_context.batch_input_shape
    batch_label_shape = global_unet_context.batch_label_shape
    raster_shape = global_unet_context.raster_shape

    raster_patches_shm = shared_memory.SharedMemory(name=global_raster_sm_name)
    anno_boun_patches_shm = shared_memory.SharedMemory(name=global_anno_boun_sm_name)

    batch_raster_shm = shared_memory.SharedMemory(name=global_raster_batch_sm_name)
    batch_anno_shm = shared_memory.SharedMemory(name=global_anno_boun_batch_sm_name)

    tf_batch_indices_sm = shared_memory.SharedMemory(name=global_tf_indices_sm_name)

    all_raster_patches = ndarray(global_shared_raster_shape, dtype=float32, buffer=raster_patches_shm.buf)
    all_anno_boun_patches = ndarray(global_shared_anno_boun_shape, dtype=uint8, buffer=anno_boun_patches_shm.buf)

    shared_raster_batch = ndarray(global_shared_raster_batch_shape, dtype=float32, buffer=batch_raster_shm.buf)
    shared_anno_boun_batch = ndarray(global_shared_anno_boun_batch_shape, dtype=uint8, buffer=batch_anno_shm.buf)

    shared_tf_indices_batch = ndarray(global_batch_queue_depth * global_batches_per_run * batch_size, dtype=uint32, buffer=tf_batch_indices_sm.buf)

    assert all_raster_patches.shape[0] == all_anno_boun_patches.shape[0]

    raster_patches = all_raster_patches[:global_training_frames_count]
    anno_boun_patches = all_anno_boun_patches[:global_training_frames_count]

    raster_empty_patches = all_raster_patches[global_training_frames_count + global_validation_frames_count :][: global_training_empty_frames_count]

    rng = default_rng()
    randint = rng.integers
    random = rng.random
    std_nrm = rng.standard_normal

    raster_xy_size = raster_shape[0]
    batch_xy_size = batch_input_shape[0]

    rnd_count = 128

    patches_per_run = batch_size
    empty_patches_this_idx = randint(0, 10, rnd_count, dtype=uint8)
    patch_offsets = randint(0, raster_xy_size - batch_xy_size, (rnd_count, 2), dtype=uint16)
    flip_xy_yes_no = randint(0, 2, (rnd_count, 2), dtype=uint8)
    standardize_patch_yes_no = randint(0, 4, rnd_count, dtype=uint8)
    roll_offsets = randint(0, batch_xy_size, (rnd_count, 2), dtype=uint16)
    roll_xy_yes_no = randint(0, 2, rnd_count, dtype=uint8)
    rotate_yes_no = randint(0, 3, rnd_count, dtype=uint8)
    scale_yes_no = randint(0, 3, rnd_count, dtype=uint8)
    skew_yes_no = randint(0, 3, rnd_count, dtype=uint8)
    affine_yes_no = randint(0, 3, rnd_count, dtype=uint8)
    uniform_noise_yes_no = randint(0, 100, rnd_count, dtype=uint8)
    gaussian_noise_yes_no = randint(0, 100, rnd_count, dtype=uint8)

    patch_indices = randint(0, raster_patches.shape[0], rnd_count, dtype=uint16)
    if raster_empty_patches.shape[0] > 0:
        empty_patch_indices = randint(0, raster_empty_patches.shape[0], rnd_count, dtype=uint16)
    else:
        empty_patch_indices = None

    tf_indices = zeros(patches_per_run, dtype=uint32)

    xy_scale = random((rnd_count, 2), dtype=float32)
    xy_skew = random((rnd_count, 2), dtype=float32)

    raster_batch = zeros((patches_per_run, *batch_input_shape), dtype=float32)
    anno_boun_batch = zeros((patches_per_run, *batch_label_shape), dtype=uint8)

    gaussian_noise = std_nrm((batch_input_shape[0], batch_input_shape[1], batch_input_shape[2]), dtype=float32)
    
    random_buffer = zeros((1, 1, 1), dtype=float32)
    random(out=random_buffer, dtype=float32)

    rotation_matrix = zeros((3, 3), dtype=float32)
    scale_matrix = zeros((3, 3), dtype=float32)
    skew_matrix = zeros((3, 3), dtype=float32)
    transform_matrix = zeros((3, 3), dtype=float32)
    identity_matrix = zeros((3, 3), dtype=float32)

    identity_matrix[0, 0] = 1
    identity_matrix[1, 1] = 1
    identity_matrix[2, 2] = 1

    scale_matrix[:] = identity_matrix
    rotation_matrix[:] = identity_matrix
    skew_matrix[:] = identity_matrix
    transform_matrix[:] = identity_matrix

    morph_yes_no = randint(0, 200, rnd_count, dtype=uint8)
    rect_matrix = getStructuringElement(MORPH_RECT, (3,3))
    ellipse_matrix = getStructuringElement(MORPH_ELLIPSE, (3,3))

    empty_anno_bound = zeros(anno_boun_patches.shape[1:], dtype=uint8)

    maximum_empty_patch_regions_count_per_map = max(1, batch_size // 16)

    half_raster_xy_size = raster_xy_size // 2

    repeat_patched_count = 1

    empty_patch_regions_count = 0
    rnd_idx = 0
    batch_idx = 0
    while batch_idx < patches_per_run:
        collect()

        select_new_patch = batch_idx % max(1, repeat_patched_count) == 0
        if select_new_patch:
            patch_idx = patch_indices[rnd_idx]
            raster = raster_patches[patch_idx]
            anno_boun = anno_boun_patches[patch_idx]
        
            empty_patch_this_idx = False
            if empty_patch_indices is not None:
                empty_patch_this_idx = empty_patches_this_idx[rnd_idx] == 0 
                if empty_patch_this_idx:
                    if empty_patch_regions_count >= maximum_empty_patch_regions_count_per_map:
                        empty_patch_this_idx = False
                    else:
                        #choice_fps = raster_empty_fps

                        empty_patch_regions_count += repeat_patched_count

                        patch_idx = empty_patch_indices[rnd_idx]
                        raster = raster_empty_patches[patch_idx]
                        anno_boun = empty_anno_bound

        attempt_count = 0
        while True:
            (y, x) = patch_offsets[rnd_idx]

            affine = affine_yes_no[rnd_idx] == 0
            if affine:
                rotate = rotate_yes_no[rnd_idx] == 0
                if rotate:
                    y = uint16(half_raster_xy_size * 0.25 + randint(-32, 32, 1, dtype=int8)[0])
                    x = uint16(half_raster_xy_size * 0.25 + randint(-32, 33, 1, dtype=int8)[0])

                    transform_matrix[:2] = getRotationMatrix2D((half_raster_xy_size, half_raster_xy_size), int(random() * 45), 1)

                skew = skew_yes_no[rnd_idx] == 0
                if skew:
                    x_skew, y_skew = (2 * xy_skew[rnd_idx] - 1) * 0.15

                    skew_matrix[1, 0] = y_skew
                    skew_matrix[0, 1] = x_skew

                    transform_matrix[:] = dot(transform_matrix, skew_matrix)

                x_scale, y_scale = 1, 1
                scale = scale_yes_no[rnd_idx] == 0
                if (not skew) and scale:
                    x_scale, y_scale = 1 + .2 * xy_scale[rnd_idx]
                    
                    scale_matrix[0, 0] = x_scale
                    scale_matrix[1, 1] = y_scale

                    transform_matrix[:] = dot(transform_matrix, scale_matrix)

                warp = not array_equal(transform_matrix, identity_matrix)
                if warp:
                    warp_xy = (raster.shape[1], raster.shape[0]) #NOTE(Jesse): warp swaps X and Y indices (x, y), not numpy (y, x)
                    raster_warp = warpAffine(raster, transform_matrix[:2], warp_xy, flags=INTER_AREA)
                    anno_boun_warp = warpAffine(anno_boun, transform_matrix[:2], warp_xy, flags=INTER_NEAREST)

                    transform_matrix[:] = identity_matrix

                    if len(raster_warp.shape) < 3:
                        raster_warp = raster_warp[..., newaxis]

                    if len(anno_boun_warp.shape) < 3:
                        anno_boun_warp = anno_boun_warp[..., newaxis]

                    anno_boun_warp[anno_boun_warp > 1] = 10

                    raster = raster_warp
                    anno_boun = anno_boun_warp

            anno_boun_patch = anno_boun[y:y + batch_xy_size, x:x + batch_xy_size]
            raster_patch = raster[y:y + batch_xy_size, x:x + batch_xy_size]
            
            if morph_yes_no[rnd_idx] == 0:
                for i in range(batch_input_shape[2]):
                    kernel = ellipse_matrix if randint(0, 2, 1, dtype=uint8)[0] == 0 else rect_matrix
                    morph_type = MORPH_CLOSE if randint(0, 2, 1, dtype=uint8)[0] == 0 else MORPH_OPEN
                    raster_patch[..., i] = morphologyEx(raster_patch[..., i], morph_type, kernel)

            un = uniform_noise_yes_no[rnd_idx] == 0
            if un:
                no_data_mask = raster_batch[..., :] == 0

                u_n = randint(-127, 128, (batch_input_shape[0], batch_input_shape[1], batch_input_shape[2]), dtype=int8).astype(float32)
                standardize_inplace(u_n)

                raster_batch[:] += u_n * 0.05

                raster_batch[no_data_mask] = 0

            gn = gaussian_noise_yes_no[rnd_idx] == 0
            if (not un) and gn:
                no_data_mask = raster_batch[..., :] == 0

                raster_batch[:] += gaussian_noise * 0.05

                std_nrm(out=gaussian_noise, dtype=float32)
                
                raster_batch[:] += raster_batch * random_buffer * gaussian_noise * 0.01

                random(out=random_buffer, dtype=float32)
                std_nrm(out=gaussian_noise, dtype=float32)

                raster_batch[no_data_mask] = 0

            rnd_idx += 1
            if rnd_idx == rnd_count:
                empty_patches_this_idx[:] = randint(0, 10, rnd_count, dtype=uint8)

                morph_yes_no[:] = randint(0, 200, rnd_count, dtype=uint8)

                patch_offsets[:] = randint(0, raster_xy_size - batch_xy_size, (rnd_count, 2), dtype=uint16)
                roll_offsets[:] = randint(0, batch_xy_size, (rnd_count, 2), dtype=uint16)

                uniform_noise_yes_no[:] = randint(0, 100, rnd_count, dtype=uint8)
                gaussian_noise_yes_no[:] = randint(0, 100, rnd_count, dtype=uint8)

                standardize_patch_yes_no[:] = randint(0, 8, rnd_count, dtype=uint8)
                roll_xy_yes_no[:] = randint(0, 2, rnd_count, dtype=uint8)
                flip_xy_yes_no[:] = randint(0, 2, (rnd_count, 2), dtype=uint8)
                rotate_yes_no[:] = randint(0, 3, rnd_count, dtype=uint8)
                affine_yes_no[:] = randint(0, 4, rnd_count, dtype=uint8)
                skew_yes_no[:] = randint(0, 3, rnd_count, dtype=uint8)
                scale_yes_no[:] = randint(0, 3, rnd_count, dtype=uint8)

                xy_skew[:] = random((rnd_count, 2), dtype=float32)
                xy_scale[:] = random((rnd_count, 2), dtype=float32)

                patch_indices[:] = randint(0, raster_patches.shape[0], rnd_count, dtype=uint16)
                if empty_patch_indices is not None:
                    empty_patch_indices[:] = randint(0, raster_empty_patches.shape[0], rnd_count, dtype=uint16)

                rnd_idx = 0

            if not empty_patch_this_idx:
                #assert anno_boun[..., 0].sum() > 0
                if (anno_boun_patch == 1).sum(dtype=uint64) == 0:
                    if attempt_count > 6:
                        break

                    patch_idx = randint(0, raster_patches.shape[0], dtype=uint16)
                    raster = raster_patches[patch_idx]
                    anno_boun = anno_boun_patches[patch_idx]

                    attempt_count += 1

                    continue

            if (raster_patch.max() == raster_patch.min() == 0) or ((raster_patch[..., 0] == 0).sum() / raster_xy_size**2) > 0.1:
                patch_idx = randint(0, raster_patches.shape[0], dtype=uint16)
                raster = raster_patches[patch_idx]
                anno_boun = anno_boun_patches[patch_idx]
                empty_patch_this_idx = False

                continue

            break

        #assert not any(isnan(batch_pan_ndvi)), tf
        #assert not any(isnan(batch_anno_boun)), tf

        raster_batch[batch_idx] = raster_patch
        if standardize_patch_yes_no[rnd_idx] > 0:
            standardize_inplace(raster_batch[batch_idx])
        anno_boun_batch[batch_idx] = anno_boun_patch

        if flip_xy_yes_no[rnd_idx, 0] == 1:
            raster_batch[batch_idx] = raster_batch[batch_idx, ::-1, :]
            anno_boun_batch[batch_idx] = anno_boun_batch[batch_idx, ::-1, :]

        if flip_xy_yes_no[rnd_idx, 1] == 1:
            raster_batch[batch_idx] = raster_batch[batch_idx, :, ::-1]
            anno_boun_batch[batch_idx] = anno_boun_batch[batch_idx, :, ::-1]

        tf_indices[batch_idx] = patch_idx if not empty_patch_this_idx else global_training_frames_count + global_validation_frames_count + patch_idx
        batch_idx += 1

    shared_raster_batch[write_base_idx:write_base_idx+patches_per_run] = raster_batch
    shared_anno_boun_batch[write_base_idx:write_base_idx+patches_per_run] = anno_boun_batch
    shared_tf_indices_batch[write_base_idx:write_base_idx+patches_per_run] = tf_indices

    raster_patches_shm.close()
    anno_boun_patches_shm.close()
    batch_raster_shm.close()
    batch_anno_shm.close()
    tf_batch_indices_sm.close()


#NOTE(Jesse): Nick at NCCS suggested this as a sanity check
if user_id is not None:
    from os import system
    import signal

    def cleanup_shared_memory(sig, frame):
        system(clear_shared_memory_string)
        exit()

    signal.signal(signal.SIGINT, cleanup_shared_memory)
    signal.signal(signal.SIGTERM, cleanup_shared_memory)


from multiprocessing import shared_memory
if __name__ == "__main__":
    from multiprocessing import Pool, set_start_method
    set_start_method("spawn")

    def main():
        from time import time
        start = time() / 60

        from os import remove, cpu_count, mkdir
        from shutil import move
        from os.path import join, isdir, isfile, normpath
        from datetime import date

        global model_training_percentage, unet_context, global_shared_raster_shape, global_shared_anno_boun_shape
        global global_shared_raster_batch_shape, global_shared_anno_boun_batch_shape, global_unet_context
        global training_data_fp, model_weights_fp, global_shared_raster_count

        model_training_ratio = model_training_percentage / 100
        assert 0 < model_training_ratio <= 1

        if global_local_tmp_dir and not isdir(global_local_tmp_dir):
            mkdir(global_local_tmp_dir)

        #NOTE(Jesse): Early failure for bad inputs.
        training_data_fp = normpath(training_data_fp)
        assert isdir(training_data_fp), training_data_fp

        model_task_name = training_data_fp.split('/')[-1]

        if model_weights_fp:
            model_weights_fp = normpath(model_weights_fp)
            assert isfile(model_weights_fp), model_weights_fp

        #from json import dump
        from numpy.random import Generator, PCG64DXSM
        from glob import glob
        from gc import collect

        rng = Generator(PCG64DXSM())

        rng_seed = int(rng.integers(1_000_000_000))
        if unet_context.is_deterministic:
            rng_seed = 1

        rng = Generator(PCG64DXSM(seed=rng_seed))

        #training_files = glob(training_data_fp + training_data_glob_match)
        import os 
        training_files = glob(os.path.join(training_data_fp, training_data_glob_match))

        print(f"Base directory: {training_data_fp}")
        print(f"Glob match: {training_data_glob_match}")
        print(f"Full search string: {training_data_fp + training_data_glob_match}")
        print(f"Training files found: {training_files[:5]}")  # Print first few files to check

        training_files_count = len(training_files)
        print(f"Number of training files found: {training_files_count}")

        assert training_files_count > 0

        #NOTE(Jesse): Sort then shuffle gives us a deterministic lever to pull.
        training_files.sort()
        rng.shuffle(training_files)

        #training_files = training_files[:256]
        #training_files_count = 256

        print(f"Loading {training_files_count} items of training data from match: {training_data_fp + training_data_glob_match}")

        frames_count = 0
        empty_frames_count = 0

        training_fps = None

        batch_size = unet_context.batch_size
        raster_shape = unet_context.raster_shape
        label_shape = unet_context.label_shape
        batch_input_shape = unet_context.batch_input_shape
        batch_label_shape = unet_context.batch_label_shape

        global_shared_raster_batch_shape = (global_batch_queue_depth * global_batches_per_run * batch_size, *batch_input_shape)
        global_shared_anno_boun_batch_shape = (global_batch_queue_depth * global_batches_per_run * batch_size, *batch_label_shape)

        global_shared_raster_shape = (training_files_count, *raster_shape)
        global_shared_anno_boun_shape = (training_files_count, *label_shape)

        global_shared_raster_shape_bytes = prod(global_shared_raster_shape) * 4
        global_shared_anno_boun_shape_bytes = prod(global_shared_anno_boun_shape) * 1

        try:
            raster_sm = shared_memory.SharedMemory(name=global_raster_sm_name)
            if raster_sm.size != global_shared_raster_shape_bytes:
                #raster_sm.close()
                raster_sm.unlink()
                raise Exception
        except Exception as e:
            raster_sm = shared_memory.SharedMemory(global_raster_sm_name, True, global_shared_raster_shape_bytes)

        try:
            anno_boun_sm = shared_memory.SharedMemory(name=global_anno_boun_sm_name)
            if anno_boun_sm.size != global_shared_anno_boun_shape_bytes:
                #anno_boun_sm.close()
                anno_boun_sm.unlink()
                raise Exception
        except Exception as e:
            anno_boun_sm = shared_memory.SharedMemory(global_anno_boun_sm_name, True, global_shared_anno_boun_shape_bytes)

        try:
            raster_batch_sm = shared_memory.SharedMemory(name=global_raster_batch_sm_name)
            if raster_batch_sm.size != prod(global_shared_raster_batch_shape) * 4:
                #raster_batch_sm.close()
                raster_batch_sm.unlink()
                raise Exception
        except Exception as e:
            raster_batch_sm = shared_memory.SharedMemory(global_raster_batch_sm_name, True, prod(global_shared_raster_batch_shape) * 4)

        try:
            anno_batch_sm =  shared_memory.SharedMemory(name=global_anno_boun_batch_sm_name)
            if anno_batch_sm.size != prod(global_shared_anno_boun_batch_shape) * 4:
                #anno_batch_sm.close()
                anno_batch_sm.unlink()
                raise Exception
        except Exception as e:
            anno_batch_sm = shared_memory.SharedMemory(global_anno_boun_batch_sm_name, True, prod(global_shared_anno_boun_batch_shape) * 4)

        try:
            tf_batch_indices_sm = shared_memory.SharedMemory(name=global_tf_indices_sm_name)
            if tf_batch_indices_sm.size != global_batch_queue_depth * global_batches_per_run * batch_size * 4:
                #tf_batch_indices_sm.close()
                tf_batch_indices_sm.unlink()
                raise Exception
        except Exception as e:
            tf_batch_indices_sm = shared_memory.SharedMemory(global_tf_indices_sm_name, True, global_batch_queue_depth * global_batches_per_run * batch_size * 4)

        shared_raster = ndarray(global_shared_raster_shape, dtype=float32, buffer=raster_sm.buf)
        assert shared_raster is not None

        shared_anno_boun = ndarray(global_shared_anno_boun_shape, dtype=uint8, buffer=anno_boun_sm.buf)
        assert shared_anno_boun is not None

        #NOTE(Jesse): Astonishingly, Python pools will automatically unlink any shared memory objects accessed by pool methods on de-init.
        # This is going to be fixed in 3.13, so until then, we need to scope the entire region.
        p1 = Pool(initializer=init_load_raster_global_variables, initargs=(unet_context, global_shared_raster_shape, global_shared_anno_boun_shape))
        try:
            out_data = p1.starmap(process_load_raster_and_anno_files, zip(training_files, range(len(training_files))), chunksize=1)
        except Exception as e:
            print(e)
            return

        training_fps = [None] * len(training_files)# for idx, is_empty in out_data if not is_empty]
        empty_frames_count = len([None for is_empty in out_data if is_empty])
        raster_count = len(training_files) - empty_frames_count
        if empty_frames_count > 0:
            tmp_r_arr = zeros(unet_context.raster_shape, dtype=float32)
            tmp_ab_arr = zeros(unet_context.label_shape, dtype=uint8)

            end_idx = shared_raster.shape[0] - 1
            assert end_idx >= 0

            end_r_slice = shared_raster[end_idx]
            end_ab_slice = shared_anno_boun[end_idx]

            for start_idx, is_empty in enumerate(out_data):
                if start_idx >= end_idx:
                    training_fps[start_idx] = training_files[start_idx]
                    break

                if not is_empty:
                    training_fps[start_idx] = training_files[start_idx]
                    continue

                while out_data[end_idx]: #is end empty
                    training_fps[end_idx] = training_files[end_idx]
                    end_idx -= 1
                    if end_idx <= start_idx:
                        break
                    
                if end_idx <= start_idx:
                    break

                #end_idx is now not empty, so swap

                start_r_slice = shared_raster[start_idx]
                start_ab_slice = shared_anno_boun[start_idx] #NOTE(Jesse): Always the same contents, as is empty, so we could optimize this a tad.

                end_r_slice = shared_raster[end_idx]
                end_ab_slice = shared_anno_boun[end_idx]

                #assert start_ab_slice.max() == 0
                #assert end_ab_slice.max() > 0

                tmp_r_arr[:] = end_r_slice
                tmp_ab_arr[:] = end_ab_slice

                end_r_slice[:] = start_r_slice[:]
                end_ab_slice[:] = start_ab_slice[:]

                start_r_slice[:] = tmp_r_arr
                start_ab_slice[:] = tmp_ab_arr

                training_fps[start_idx] = training_files[end_idx]
                training_fps[end_idx] = training_files[start_idx]
                
                end_idx -= 1
                assert end_idx >= 0
                assert end_idx >= start_idx

            tmp_r_arr = None
            tmp_ab_arr = None

        out_data = None

        global_shared_raster_count = raster_count
        frames_count = global_shared_raster_count

        collect()

        print(f"Found {frames_count} labeled regions and {empty_frames_count} empty regions")
        
        shared_raster_batch = ndarray(global_shared_raster_batch_shape, dtype=float32, buffer=raster_batch_sm.buf)
        shared_anno_boun_batch = ndarray(global_shared_anno_boun_batch_shape, dtype=uint8, buffer=anno_batch_sm.buf)

        tf_batch_indices = ndarray(global_batch_queue_depth * global_batches_per_run * batch_size, dtype=uint32, buffer=tf_batch_indices_sm.buf)

        collect()

        training_ratio = 0.9 * model_training_ratio
        validation_ratio = 1 - training_ratio
        assert .999 < training_ratio + validation_ratio <= 1

        training_frames_count = int((frames_count * training_ratio))
        training_empty_frames_count = int((empty_frames_count * training_ratio))

        validation_frames_count = int((frames_count * validation_ratio))
        validation_empty_frames_count = int((empty_frames_count * validation_ratio))

        print(f"Training ratio: {int(training_ratio * 100)}%, Validation ratio: {int(validation_ratio * 100)}%")
        print(f"Training label count: {training_frames_count}, Validation label count: {validation_frames_count}")
        print(f"Empty Training label count: {training_empty_frames_count}, Empty Validation label count: {validation_empty_frames_count}")
        
        debug_sort = True

        training_raster_frames = shared_raster[: training_frames_count]
        training_anno_boun_frames = shared_anno_boun[: training_frames_count]
        if debug_sort:
            for i in range(training_frames_count):
                assert training_anno_boun_frames[i].max() > 0

        validation_raster_frames = shared_raster[training_frames_count :][: validation_frames_count]
        validation_anno_boun_frames = shared_anno_boun[training_frames_count :][: validation_frames_count]
        if debug_sort:
            for i in range(validation_frames_count):
                assert validation_anno_boun_frames[i].max() > 0

        training_raster_empty_frames = shared_raster[training_frames_count + validation_frames_count :][: training_empty_frames_count]
        validation_raster_empty_frames = shared_raster[training_frames_count + validation_frames_count + training_empty_frames_count:]
        if debug_sort:
            validation_empty_ab = shared_anno_boun[training_frames_count + validation_frames_count + training_empty_frames_count:]
            for i in range(validation_empty_frames_count):
                assert validation_empty_ab[i].max() == 0

        assert frames_count - 1 <= training_raster_frames.shape[0] + validation_raster_frames.shape[0] <= frames_count + 1
        assert empty_frames_count -1 <= training_raster_empty_frames.shape[0] + validation_raster_empty_frames.shape[0] <= empty_frames_count + 1
        
        training_raster_fps = training_fps[: training_frames_count]
        training_raster_empty_fps = training_fps[training_frames_count + validation_frames_count :][: training_empty_frames_count]

        pool_args = (global_shared_raster_shape, global_shared_anno_boun_shape, global_shared_raster_batch_shape, global_shared_anno_boun_batch_shape, global_shared_raster_count, unet_context, training_frames_count, training_empty_frames_count, validation_frames_count, validation_empty_frames_count)

        debug = True
        p2 = None
        if debug:
            p2 = Pool(10, initializer=init_pool_global_variables, initargs=pool_args)
            from unet.visualize import display_images
            batch_queue_index = 0
            while True:
                batch_base_index = batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)
                overlapped_batch_waitable = p2.map_async(process_uniform_with_data_aug_random_patch, range(batch_base_index, batch_base_index + global_batches_per_run * batch_size, batch_size))
                overlapped_batch_waitable.wait()
                assert overlapped_batch_waitable.successful()

                for batch_idx in range(global_batches_per_run):
                    raster_batch = shared_raster_batch[batch_base_index +  batch_idx * batch_size : batch_base_index + (batch_idx + 1) * batch_size]
                    anno_batch = shared_anno_boun_batch[batch_base_index + batch_idx * batch_size : batch_base_index + (batch_idx + 1) * batch_size]

                    tf_batch = tf_batch_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                    tf_fps = [training_fps[idx] for idx in tf_batch]

                    ann = anno_batch == 1
                    wei = (anno_batch == 10) * 10
                    wei[wei == 0] = 1
                    display_images(concatenate((raster_batch, ann, ann + wei), axis=-1), training_data_fp)
                    bob = True

                batch_queue_index = (batch_queue_index + 1) % global_batch_queue_depth

        print("Loading Model API")
        environ["KERAS_BACKEND"] = "jax"

        #NOTE(Jesse): JAX uses these TF_ env variables
        environ["TF_GPU_THREAD_MODE"] = "gpu_private" #NOTE(Jesse): Seperate I/O and Compute CPU thread scheduling.
        environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

        #from keras.callbacks import ModelCheckpoint#, TensorBoard
        from unet.loss import focal_tversky, tversky, accuracy, dice_coef
        import keras as K
        from keras.optimizers import SGD, AdamW, schedules#, Nadam, Adadelta, schedules

        cdr_ls = schedules.CosineDecayRestarts

        UNet_version = unet_context.UNet_version
        raster_bands = unet_context.raster_bands

        K.utils.clear_session(free_memory=True)

        K.utils.set_random_seed(rng_seed)
        data_parallel = K.distribution.DataParallel()
        K.distribution.set_distribution(data_parallel)

        if UNet_version == 1:
            from unet.UNet import UNet_v1 as UNet
        elif UNet_version == 2:
            from unet.UNet import UNet_v2 as UNet
        elif UNet_version == 3:
            from unet.AttnUNetMultiInput import attn_reg_train as UNet
            from unet.AttnUNetMultiInput import attn_reg
            
        model = UNet(batch_input_shape, weights_file=model_weights_fp)

        base_lr = 0.001 * global_lr_multiplier
        lr = base_lr * (unet_context.batch_size if not global_opt_adamw_toggle else 1.0)
        wd=0.004 * global_wd_multiplier
        mom1 = 0.9 * global_momentum1_multiplier
        mom2 = 0.999 * global_momentum2_multiplier

        decay_steps = steps_per_epoch
        ilr = lr

        lr_s = cdr_ls(ilr, decay_steps)

        sgd = SGD(learning_rate=lr_s if global_lr_schedule_toggle else lr, momentum=mom1, nesterov=True, weight_decay=wd)
        adamw = AdamW(learning_rate=lr_s if global_lr_schedule_toggle else lr, beta_1=mom1, beta_2=mom2, weight_decay=wd)
        optimizer = adamw if global_opt_adamw_toggle else sgd

        v3_prune_str = "_train"
        unet_weights_fn = unet_weights_fn_template.format("_".join(raster_bands) + "_" + date.today().strftime("%d-%b-%Y") + "_v-" + f"{UNet_version}" + f"_model_number-{model_number}" + f"_label_percentage-{model_training_percentage}" + f"_epoch-{epochs}" + f"_steps-{steps_per_epoch}" + "_opt-" + optimizer.name + "_" + model_task_name + (("_" + "_".join(global_opt_params)) if global_opt_params is not None else "") + (v3_prune_str if UNet_version == 3 else ""))
        print(unet_weights_fn)

        post_train = False
        if model_weights_fp:
            post_train = True
        else:
            weights_tmp_dir = training_data_fp
            if a100_partition:
                weights_tmp_dir = environ["TSE_TMPDIR"]
                assert isdir(weights_tmp_dir)

            model_weights_fp = join(weights_tmp_dir, unet_weights_fn)

        loss_cfg = tversky
        loss_str_match = "loss"
        m = [dice_coef, accuracy]
        if UNet_version == 3:
            sb2 = zeros((global_batches_per_run * batch_size, batch_label_shape[0] // 2, batch_label_shape[1] // 2, batch_label_shape[2]), dtype=float32)
            sb3 = zeros((global_batches_per_run * batch_size, batch_label_shape[0] // 4, batch_label_shape[1] // 4, batch_label_shape[2]), dtype=float32)
            sb4 = zeros((global_batches_per_run * batch_size, batch_label_shape[0] // 8, batch_label_shape[1] // 8, batch_label_shape[2]), dtype=float32)                

            m = [m]
            m *= 4

            loss_cfg = {
                'pred1':focal_tversky,
                'pred2':focal_tversky,
                'pred3':focal_tversky,
                'final': tversky
            }

            loss_str_match = "final_loss"
            
        #tensorboard = TensorBoard(log_dir=training_data_fp, histogram_freq=1)

        print("Start training")
        evaluation_steps = evaluation_count_target // batch_size
        model_fit = model.fit
        model_evaluate = model.evaluate

        previous_loss = 1
        failed_count = 0
        epoch_idx = 0 if not post_train else epochs

        p3 = Pool(max(1, cpu_count() - 2), initializer=init_pool_global_variables, initargs=pool_args)
        cpu_batch_queue_index = 0
        cpu_batch_base_index = cpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)

        gpu_batch_queue_index = 0
        gpu_batch_base_index = gpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)

        #NOTE(Jesse): Overlap next batch production with CURRENT training step
        #NOTE(Jesse): Pre-emptively produce first batch
        overlapped_batch_waitable = p3.map_async(process_uniform_with_data_aug_random_patch, range(cpu_batch_base_index, cpu_batch_base_index + global_batches_per_run * batch_size, batch_size))

        model.compile(optimizer=optimizer, loss=loss_cfg, metrics=m, jit_compile=True)

        overlapped_batch_waitable.wait()
        assert overlapped_batch_waitable.successful()

        cpu_batch_queue_index = (cpu_batch_queue_index + 1) & (global_batch_queue_depth - 1)
        cpu_batch_base_index = cpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)

        #NOTE(Jesse): gpu_batch_queue_index = do nothing here
        gpu_batch_base_index = gpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)

        while epoch_idx < epochs:
            collect()
            print(f"Begin training epoch {epoch_idx + 1}.") 
            epoch_idx += 1

            training_batches_count = steps_per_epoch // global_batches_per_run
            for s in range(training_batches_count):
                #NOTE(Jesse): Begin filling the NEXT batch as we process the CURRENT batch, except for the last time through the loop, to generate a validation batch
                proc = process_uniform_with_data_aug_random_patch if (s < training_batches_count - 1) else process_uniform_patch_generator
                overlapped_batch_waitable = p3.map_async(proc, range(cpu_batch_base_index, cpu_batch_base_index + global_batches_per_run * batch_size, batch_size))

                sb1 = shared_anno_boun_batch[gpu_batch_base_index:gpu_batch_base_index + global_batches_per_run * batch_size]
                sb = sb1
                if UNet_version == 3:
                    sb2[:] = sb1[..., ::2, ::2, :]
                    sb3[:] = sb2[..., ::2, ::2, :]
                    sb4[:] = sb3[..., ::2, ::2, :]
                    sb = [sb4, sb3, sb2, sb1]

                #NOTE(Jesse): Tried to incrementally increase batch size during training but it causes difficult to manage OOM issues, and tends to trade larger absolute batch size for variable but smaller batch sizes.
                this_batch_size = batch_size#min(2 ** (5 + s), batch_size)
                train_val_loss_and_metrics = model_fit(shared_raster_batch[gpu_batch_base_index:gpu_batch_base_index + global_batches_per_run * batch_size], sb, batch_size=this_batch_size, shuffle=False, initial_epoch=epoch_idx - 1, epochs=epoch_idx).history#, callbacks=cb).history
                #print ("\033[A                             \033[A")

                overlapped_batch_waitable.wait() #NOTE(Jesse): Presumably the CPU batch filling has already completed.
                assert overlapped_batch_waitable.successful()

                cpu_batch_queue_index = (cpu_batch_queue_index + 1) & (global_batch_queue_depth - 1)
                cpu_batch_base_index = cpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)

                gpu_batch_queue_index = (gpu_batch_queue_index + 1) & (global_batch_queue_depth - 1)
                gpu_batch_base_index = gpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)
            #print(train_val_loss_and_metrics)
            if isnan(train_val_loss_and_metrics[loss_str_match]):
                print("[ERROR] nan detected during training.  Early Stopping.")
                #remove(model_weights_fp)

                return

            print("Eval")

            eval_val_loss_and_metrics = None
            eval_batches_count = max(2, evaluation_steps // global_batches_per_run)
            for s in range(eval_batches_count):
                proc = process_uniform_patch_generator if (s < eval_batches_count - 1) else process_uniform_with_data_aug_random_patch

                overlapped_batch_waitable = p3.map_async(proc, range(cpu_batch_base_index, cpu_batch_base_index + global_batches_per_run * batch_size, batch_size))

                sb1 = shared_anno_boun_batch[gpu_batch_base_index:gpu_batch_base_index + global_batches_per_run * batch_size]
                sb = sb1
                if UNet_version == 3:
                    sb2[:] = sb1[..., ::2, ::2, :]
                    sb3[:] = sb2[..., ::2, ::2, :]
                    sb4[:] = sb3[..., ::2, ::2, :]
                    sb = [sb4, sb3, sb2, sb1]

                this_val_loss_and_metrics = model_evaluate(shared_raster_batch[gpu_batch_base_index:gpu_batch_base_index + global_batches_per_run * batch_size], sb, batch_size=batch_size, return_dict=True)
                if eval_val_loss_and_metrics:
                    for k, v in this_val_loss_and_metrics.items():
                        eval_val_loss_and_metrics[k] += v
                else:
                    eval_val_loss_and_metrics = this_val_loss_and_metrics

                overlapped_batch_waitable.wait()
                assert overlapped_batch_waitable.successful()

                cpu_batch_queue_index = (cpu_batch_queue_index + 1) & (global_batch_queue_depth - 1)
                cpu_batch_base_index = cpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)

                gpu_batch_queue_index = (gpu_batch_queue_index + 1) & (global_batch_queue_depth - 1)
                gpu_batch_base_index = gpu_batch_queue_index * (global_shared_raster_batch_shape[0] // global_batch_queue_depth)

            for k in eval_val_loss_and_metrics:
                eval_val_loss_and_metrics[k] /= eval_batches_count

            print(eval_val_loss_and_metrics)

            val_loss_this_epoch = eval_val_loss_and_metrics[loss_str_match]
            if isnan(val_loss_this_epoch) or (val_loss_this_epoch > previous_loss):
                print(f"Training epoch resulted in worse loss {val_loss_this_epoch} than before {previous_loss}.")
                failed_count += 1

                if failed_count >= 10:
                    print("Early stop training.")
                    break

                continue

            if val_loss_this_epoch < previous_loss:
                print(f"Loss reduced by {previous_loss - val_loss_this_epoch}! Now: {val_loss_this_epoch}")
                previous_loss = val_loss_this_epoch
                failed_count = 0

                model.save_weights(model_weights_fp)

                continue

        if not post_train and a100_partition:
            if isfile(join(training_data_fp, unet_weights_fn)):
                remove(join(training_data_fp, unet_weights_fn))

            move(model_weights_fp, training_data_fp)
            model_weights_fp = join(training_data_fp, unet_weights_fn)

        if UNet_version == 3:
            sb4 = None
            sb3 = None
            sb2 = None

            #NOTE(Jesse): The attn_reg model for training has extra outputs and weights that are not necessary for inferrence and incur a substantial performance cost.
            # So we "prune" them here.
            #
            # Also, TF / Keras globally namespace layer names, so if two identical models are created, they _do not_ share the same layer names
            # if layer names are not explicit provided.  This is stupid and causes all this dumb code to exist for no reason.
            # These models have over a hundred layers and they continue to grow so I don't think it's reasonable to solve it on a per
            # layer basis
            K.utils.clear_session(free_memory=True)
            
            K.utils.set_random_seed(rng_seed)
            model = UNet(batch_input_shape, weights_file=model_weights_fp)

            K.utils.set_random_seed(rng_seed)
            dst_model = attn_reg(batch_input_shape)

            blacklisted_lyrs = ("pred1", "pred2", "pred3", "conv6", "conv7", "conv8")
            base_idx = 0
            for lyr_idx, dst_lyr in enumerate(dst_model.layers):
                while True:
                    src_lyr = model.get_layer(index=lyr_idx + base_idx)
                    for bl_lyrn in blacklisted_lyrs:
                        if src_lyr.name.startswith(bl_lyrn):
                            base_idx += 1
                            src_lyr = model.get_layer(index=lyr_idx + base_idx)
                            break
                    else:
                        break

                src_lyr_wghts = src_lyr.get_weights()
                if len(src_lyr_wghts) == 0:
                    continue

                assert src_lyr.output.shape == dst_lyr.output.shape
                
                dst_lyr.set_weights(src_lyr_wghts)

            unet_weights_fp = join(training_data_fp, unet_weights_fn.replace(v3_prune_str, ""))
            dst_model.save_weights(unet_weights_fp)
            dst_model = None

        model = None

        shared_raster = None
        shared_raster_batch = None
        
        shared_anno_boun = None
        shared_anno_boun_batch = None

        try:
            #raster_sm.close()
            raster_sm.unlink()

            #anno_boun_sm.close()
            anno_boun_sm.unlink()

            #raster_batch_sm.close()
            raster_batch_sm.unlink()

            #anno_batch_sm.close()
            anno_batch_sm.unlink()

            #tf_batch_indices_sm.close()
            tf_batch_indices_sm.unlink()
        except Exception as e:
            print(e)

        p1.close()
        p1.join()

        if p2:
            p2.close()
            p2.join()

        p3.close()
        p3.join()

        stop = time() / 60
        print(f"Took {stop - start} minutes to train {unet_weights_fn}.")

    unet_context = unet_config()

    from sys import argv
    argc = len(argv)
    if argc >= 3:
        training_data_fp = argv[1]
        training_data_glob_match = argv[2]

        if argc >= 5:
            model_number = int(argv[3])
            model_training_percentage = int(argv[4])

            if argc >= 6:
                unet_context.set_raster_bands(tuple(argv[5].split(",")))

                if argc >= 7:
                    opt_params = tuple(map(int, argv[6].split(",")))
                    assert len(opt_params) == 6

                    global_opt_params = (str(x) for x in opt_params)

                    global_lr_multiplier = opt_params[0] / 1000.0
                    global_momentum1_multiplier = opt_params[1] / 100.0
                    global_momentum2_multiplier = opt_params[2] / 100.0
                    global_wd_multiplier = opt_params[3] / 100.0
                    global_lr_schedule_toggle = bool(opt_params[4])
                    global_opt_adamw_toggle = bool(opt_params[5])

    system(clear_shared_memory_string)
    main()

    if user_id is not None:
        system(clear_shared_memory_string)
