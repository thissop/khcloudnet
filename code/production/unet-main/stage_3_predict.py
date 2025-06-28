#NOTE(Jesse): The premise is to apply a trained CNN to detect tree canopies in the provided mosaic tile.
# The outputs are a raster and vector dataset of the predicted shadow geometries.

#NOTE(Jesse): The format of the input arguments are:
# argv[1]: A list of tile file paths to infer over
# argv[2]: The destination file path where results are placed
# argv[3]: File path to model weights.  The script will parse the file name to configure the UNET constructed by the script.

out_fp = "/path/to/outputs/"
model_weights_fp = "/path/to/weights/w.h5"

label_name = "trees" #NOTE(Jesse): What is being predicted?
vector_simplification_amount = 0.5 #NOTE(Jesse): Are predicted vector geometries to be simplified?  Set to > 0 to enable, in accordance with OGR Geometry Simplify.
smallest_geometry_size = 5 #NOTE(Jesse): Square meters

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

from osgeo import gdal, ogr
from concurrent import futures

out_gpkg_tmp_fn = f"{label_name}_tmp.gpkg"

disk_create_options: list = ["COMPRESS=DEFLATE", "INTERLEAVE=BAND", "Tiled=YES", "NUM_THREADS=ALL_CPUS", "SPARSE_OK=True"]

def tile_read(inputs):
    from os.path import normpath, isfile, isdir, join
    from os import mkdir, remove
    from shutil import copy
    from numpy import array, uint16, zeros

    tile_fp, local_tmp_dir, batch_input_shape, raster_band_indices = inputs
    
    print(tile_fp)

    tile_fp = normpath(tile_fp)

    assert isfile(tile_fp), tile_fp 
    assert isdir(out_fp), out_fp

    tile_fn = tile_fp.split('/')[-1].split('.')[0]

    zip_tmp_fn = f"{tile_fn}_{label_name}_tmp.zip"

    zip_fn = zip_tmp_fn.replace("_tmp", "")
    zip_fp = join(out_fp, zip_fn)
    if isfile(zip_fp):
        print(f"Result file {zip_fp} already exists!")
        return None, None, None, None, None, None, None, None, None, None, None

    tmp_dir = join(local_tmp_dir if local_tmp_dir is not None else out_fp, tile_fn)
    if not isdir(tmp_dir):
        mkdir(tmp_dir)

    if local_tmp_dir:
        tile_fp = copy(tile_fp, tmp_dir)

    zip_tmp_fp = join(tmp_dir, zip_tmp_fn)

    tile_ds = gdal.Open(tile_fp)
    tile_x = tile_ds.RasterXSize
    tile_y = tile_ds.RasterYSize

    raster_meters_per_texel = 0.5
    density_meters_per_texel = 100
    density_x_texel_count = int(((density_meters_per_texel - 1) + tile_x * raster_meters_per_texel) / density_meters_per_texel)
    density_y_texel_count = int(((density_meters_per_texel - 1) + tile_y * raster_meters_per_texel) / density_meters_per_texel)
    if tile_fp.endswith("mosaic.tif"):
        if tile_x != 32768:
            assert tile_x == 32000, tile_fp
            assert density_x_texel_count == 160, density_x_texel_count
        else:
            assert density_x_texel_count == 164, density_x_texel_count

        assert density_x_texel_count == density_y_texel_count, (density_x_texel_count, density_y_texel_count)
        assert tile_x == tile_y, tile_fp

    geotransform = tile_ds.GetGeoTransform()
    out_projection = tile_ds.GetProjection()

    density_gxfm = array(geotransform)
    density_gxfm[1] = density_meters_per_texel
    density_gxfm[5] = -density_meters_per_texel

    print("Read tile bands")
    raster = zeros((tile_ds.RasterYSize, tile_ds.RasterXSize, batch_input_shape[-1]), uint16)
    for idx, i in enumerate(raster_band_indices):
        tile_ds.GetRasterBand(i).ReadAsArray(buf_obj=raster[..., idx])

    tile_ds = None

    if local_tmp_dir:
        remove(tile_fp)

    return raster, tile_x, tile_y, geotransform, out_projection, density_x_texel_count, density_y_texel_count, density_gxfm, tmp_dir, zip_tmp_fp, zip_fp


def generate_derived_products(inputs):
    import sozipfile.sozipfile as sozipfile
    from os.path import join
    from shutil import move, rmtree
    from time import time
    from os import environ

    environ["OGR_SQLITE_PRAGMA"] = "threads=4" #NOTE(Jesse): SQLite can generate indices in parallel

    tile_start_seconds, tile_x, tile_y, no_data_value, geotransform, out_projection, out_predictions, density_x_texel_count, density_y_texel_count, density_gxfm, tmp_dir, zip_tmp_fp, zip_fp = inputs

    nn_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=tile_x, ysize=tile_y, bands=1, eType=gdal.GDT_Byte)
    assert nn_mem_ds

    nn_mem_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    nn_mem_ds.SetGeoTransform(geotransform)
    nn_mem_ds.SetProjection(out_projection)
    nn_mem_band = nn_mem_ds.GetRasterBand(1)

    nn_mem_band.WriteArray(out_predictions)

    print("Creating prediction mask")
    #NOTE(Jesse): Convert raster predictions to vector geometries via binary mask of >50% thresholding
    arr = out_predictions.copy()

    arr[arr == no_data_value] = 0
    arr[arr <= 50] = 0
    arr[arr > 50] = 1
    nn_mem_band.WriteArray(arr)
    arr = None

    src_ogr_mem_ds = gdal.GetDriverByName("Memory").Create('', 0, 0, 0, gdal.GDT_Unknown)#NOTE(Jesse): In gdal 3.10 the following will work: ogr.GetDriverByName("Memory").CreateDataSource("")

    src_ogr_mem_lyr = src_ogr_mem_ds.CreateLayer(label_name, nn_mem_ds.GetSpatialRef(), ogr.wkbPolygon)

    print("Polygonize")
    gdal.Polygonize(nn_mem_band, nn_mem_band, src_ogr_mem_lyr, -1)

    nn_mem_band.WriteArray(out_predictions)
    out_predictions = None

    feature_count = src_ogr_mem_lyr.GetFeatureCount()
    print(f"{feature_count} {label_name} predicted.")

    #NOTE(Jesse): Running these steps in parallel did not increase performance, due to contention somewhere in the Python -> GDAL -> SQLite stack.

    ogr_mem_ds = src_ogr_mem_ds
    geo_lyr = src_ogr_mem_lyr

    if vector_simplification_amount > 0:
        print(f"Simplify to {vector_simplification_amount} and cull to {smallest_geometry_size}")

        sql_stmt = f'select SimplifyPreserveTopology(GEOMETRY, {vector_simplification_amount}) from {label_name}'
        if smallest_geometry_size > 0:
            sql_stmt += f' where ST_Area(GEOMETRY) > {smallest_geometry_size}'

        geo_lyr = src_ogr_mem_ds.ExecuteSQL(sql_stmt, dialect="SQLITE")

        ogr_mem_ds = gdal.GetDriverByName("Memory").Create('', 0, 0, 0, gdal.GDT_Unknown)#ogr.GetDriverByName("Memory").CreateDataSource("")
        ogr_mem_ds.CopyLayer(geo_lyr, label_name)

        src_ogr_mem_ds.ReleaseResultSet(geo_lyr)
        src_ogr_mem_lyr = None
        src_ogr_mem_ds = None

        geo_lyr = ogr_mem_ds.GetLayer(0)

    print("Count raster")
    cnt_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=density_x_texel_count, ysize=density_y_texel_count, bands=1, eType=gdal.GDT_UInt16)

    cnt_mem_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    cnt_mem_ds.SetGeoTransform(density_gxfm)
    cnt_mem_ds.SetProjection(out_projection)

    ro = gdal.RasterizeOptions(bands=[1], burnValues=1, SQLStatement=f'select ST_Centroid(GEOMETRY) from {label_name}', SQLDialect="SQLITE", allTouched=False, add=True)
    gdal.Rasterize(cnt_mem_ds, ogr_mem_ds, options=ro)

    print("Area Raster")
    area_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=density_x_texel_count, ysize=density_y_texel_count, bands=1, eType=gdal.GDT_Float32)

    area_mem_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    area_mem_ds.SetGeoTransform(density_gxfm)
    area_mem_ds.SetProjection(out_projection)

    ro = gdal.RasterizeOptions(bands=[1], attribute="area", SQLStatement=f'select ST_Centroid(GEOMETRY), ST_Area(GEOMETRY) as area from {label_name}', SQLDialect="SQLITE", allTouched=False, add=True)
    gdal.Rasterize(area_mem_ds, ogr_mem_ds, options=ro)

    print("Save out")
    out_gpkg_tmp_fn = f"{label_name}_tmp.gpkg"
    ogr_dsk_ds = ogr.GetDriverByName("GPKG").CreateDataSource(join(tmp_dir, out_gpkg_tmp_fn)) #ogr.GetDriverByName("GPKG").CopyDataSource(ogr_mem_ds, join(tmp_dir, out_tmp_fn))
    ogr_dsk_ds.CopyLayer(geo_lyr, label_name)

    geo_lyr = None
    ogr_mem_ds = None
    ogr_dsk_ds = None

    disk_create_options: list = ["COMPRESS=DEFLATE", "INTERLEAVE=BAND", "Tiled=YES", "NUM_THREADS=ALL_CPUS", "SPARSE_OK=True"]
    nn_disk_ds = gdal.GetDriverByName("GTiff").CreateCopy(join(tmp_dir, "NN_classification_tmp.tif"), nn_mem_ds, options=disk_create_options)
    nn_cnt_ds = gdal.GetDriverByName("GTiff").CreateCopy(join(tmp_dir, f"{label_name}_count_tmp.tif"), cnt_mem_ds, options=disk_create_options)
    nn_area_ds = gdal.GetDriverByName("GTiff").CreateCopy(join(tmp_dir, f"{label_name}_area_tmp.tif"), area_mem_ds, options=disk_create_options)

    nn_mem_ds = None
    cnt_mem_ds = None
    area_mem_ds = None

    nn_disk_ds = None
    nn_cnt_ds = None
    nn_area_ds = None

    with sozipfile.ZipFile(zip_tmp_fp, 'w', compression=sozipfile.ZIP_DEFLATED) as myzip:
        myzip.write(join(tmp_dir, out_gpkg_tmp_fn), arcname=out_gpkg_tmp_fn.replace("_tmp", ""))
        myzip.write(join(tmp_dir, "NN_classification_tmp.tif"), arcname="NN_classification.tif", compress_type=None)
        myzip.write(join(tmp_dir, f"{label_name}_count_tmp.tif"), arcname=f"{label_name}_count.tif", compress_type=None)
        myzip.write(join(tmp_dir, f"{label_name}_area_tmp.tif"), arcname=f"{label_name}_area.tif", compress_type=None)

    move(zip_tmp_fp, zip_fp)

    rmtree(tmp_dir)

    print(f"Tile took {(time() - tile_start_seconds) / 60} minutes")

from unet.config import unet_config
unet_ctx = unet_config()

def main(tile_fps):
    print("Import")
    from time import time
    main_start = time()

    from os import environ
    from os.path import isfile, normpath
    from predict_utilities import predict
    from gc import collect

    gdal.UseExceptions()
    ogr.UseExceptions()

    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

    gdal.SetConfigOption("OGR_SQLITE_SYNCHRONOUS", "OFF")
    gdal.SetConfigOption("OGR_SQLITE_CACHE", "1024")
    gdal.SetConfigOption("OGR_SQLITE_JOURNAL", "OFF")

    environ["KERAS_BACKEND"] = "jax"
    environ["TF_GPU_THREAD_MODE"] = "gpu_private" #NOTE(Jesse): Seperate I/O and Compute CPU thread scheduling.
    environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    global model_weights_fp
    global out_fp
    global label_name
    global vector_simplification_amount

    assert label_name is not None
    assert vector_simplification_amount is not None
    assert isinstance(vector_simplification_amount, (int, float))

    local_tmp_dir = environ.get("LOCAL_TMPDIR")
    if local_tmp_dir is None:
        local_tmp_dir = environ.get("TSE_TMPDIR")

    print("UNET")

    model_weights_fp = normpath(model_weights_fp)
    assert isfile(model_weights_fp), model_weights_fp

    UNet_version = unet_ctx.UNet_version
    batch_input_shape = unet_ctx.batch_input_shape

    if UNet_version == 1:
        from unet.UNet import UNet_v1 as UNet
    elif UNet_version == 2:
        from unet.UNet import UNet_v2 as UNet
    elif UNet_version == 3:
        from unet.AttnUNetMultiInput import attn_reg as UNet

    model = UNet(unet_ctx.batch_input_shape, weights_file=model_weights_fp)
    model.compile(jit_compile=True)

    tile_idx = 0
    tile_read_executor = futures.ThreadPoolExecutor(1)
    tr_future = tile_read_executor.submit(tile_read, (tile_fps[tile_idx], local_tmp_dir, batch_input_shape, unet_ctx.raster_band_indices))

    derived_executor = futures.ThreadPoolExecutor(4)
    derived_work_futures = set()

    while tr_future is not None:
        tile_start_seconds = time()

        raster, tile_x, tile_y, geotransform, out_projection, density_x_texel_count, density_y_texel_count, density_gxfm, tmp_dir, zip_tmp_fp, zip_fp = tr_future.result()

        tile_idx += 1
        tr_future = tile_read_executor.submit(tile_read, (tile_fps[tile_idx], local_tmp_dir, batch_input_shape, unet_ctx.raster_band_indices)) if tile_idx < len(tile_fps) else None

        if raster is None:
            continue

        print("Predict")
        out_predictions = predict(model.predict_on_batch, raster, unet_ctx)
        a = time()
        print(f"Took {(a - tile_start_seconds) / 60} minutes to predict.")

        no_data_value = 255
        out_predictions[raster[..., 0] == 0] = no_data_value #NOTE(Jesse): Transfer no-data value from mosaic tile to these results.

        raster = None

        collect()

        if len(derived_work_futures) == 4:
            _, derived_work_futures = futures.wait(derived_work_futures, return_when=futures.FIRST_COMPLETED)

        inputs = (tile_start_seconds, tile_x, tile_y, no_data_value, geotransform, out_projection, out_predictions, density_x_texel_count, density_y_texel_count, density_gxfm, tmp_dir, zip_tmp_fp, zip_fp)
        derived_work_futures.add(derived_executor.submit(generate_derived_products, inputs))

    tile_read_executor.shutdown(wait=True)
    derived_executor.shutdown(wait=True)
    print(f"All tiles took {(time() - main_start) / 60} minutes")

#from glob import glob
#tile_fps = glob("/discover/nobackup/projects/setsm_tucker/ppl/cander15/arctic/aoi6_test_labels/" + "*/*/*.tif")
#main(tile_fps)

from sys import argv

argv_len = len(argv)
assert argv_len <= 4

if argv_len >= 2:
    tile_fps = argv[1]

    if argv_len >= 3:
        out_fp = argv[2]

        if argv_len == 4:
            model_weights_fp = argv[3]
            #model_weights_fp = "/path/to/unet_mc_p_g_r_v_05-Nov-2024_v-3_model_number-51_label_percentage-75_epoch-10_steps-1024_opt-adamw_combined_512x512.weights.h5"
            model_weights_fn = model_weights_fp.split(".")[0].split("/")[-1]
            model_weights_parts = model_weights_fn.split("_")

            first_band_idx = 2 #NOTE(Jesse): "p" in the example above
            date_idx = first_band_idx
            for p in model_weights_parts[first_band_idx:]:
                if "-" in p: #NOTE(Jesse): The first item in the unet model filename with a - is the date.
                    break

                date_idx += 1

            bands = tuple(model_weights_parts[first_band_idx:date_idx])
            unet_ctx.set_raster_bands(bands)

            if "v-3" in model_weights_fp:
                unet_ctx.UNet_version = 3
            elif "v-2" in model_weights_fp:
                unet_ctx.UNet_version = 2
            else:
                unet_ctx.UNet_version = 1                

main(tile_fps.split("\n") if "\n" in tile_fps else [tile_fps])
