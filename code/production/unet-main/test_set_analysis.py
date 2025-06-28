test_set_spring_label_fp_glob_match = "/discover/nobackup/jrmeyer3/training_data/tree_training/test_set/spring/*/*annotation_and_boundary*.tif"
test_set_summer_label_fp_glob_match = "/discover/nobackup/jrmeyer3/training_data/tree_training/test_set/summer/*/*annotation_and_boundary*.tif"
train_set_src_mosaic_tile_fp = "/discover/nobackup/jrmeyer3/arctic_grid_search_test_tiles_results/"

test_set_spring_label_fp_glob_match = "/Users/jrmeyer3/Desktop/arctic_area6/spring/*/*annotation_and_boundary.tif"
test_set_summer_label_fp_glob_match = "/Users/jrmeyer3/Desktop/arctic_area6/summer/*/*annotation_and_boundary.tif"
train_set_src_mosaic_tile_fp = "/Users/jrmeyer3/Desktop/arctic_area6/arctic_grid_search_test_tiles_results/"

time_categories = ("spring", "summer")
label_categories = ("pan", "pan_ndvi")

if __name__ != "__main__":
    exit()

#NOTE(Jesse): Losses here should stay insync with loss.py.  Specialized here due to existing outside of tensorflow ecosystem
from numpy import sum, uint64, array
def tversky(y_true, y_pred, alpha=0.75, smooth=0.0001): #NOTE(Jesse): Computes a kind of "How much information was 'lost' from the ground truth given the predictions" with a slight emphasis against false positives
    y_t = y_true

    tp = sum(y_pred * y_t, dtype=uint64)
    fp = alpha * sum(y_pred * (1 - y_t), dtype=uint64)
    fn = (1 - alpha) * sum((1 - y_pred) * y_t, dtype=uint64)

    numerator = tp
    denominator = tp + fp + fn

    if y_true.sum() == 0: #NOTE(Jesse): Empty patch handling.
        tn = sum((1 - y_pred) * (1 - y_t), dtype=uint64)
        numerator += tn
        denominator += tn

    score = (numerator + smooth) / (denominator + smooth)
    return 1 - score

def accuracy(y_true, y_pred):
    y_t = y_true

    tp = sum(y_pred * y_t, dtype=uint64)
    fp = sum(y_pred * (1 - y_t), dtype=uint64)
    fn = sum((1 - y_pred) * y_t, dtype=uint64)
    tn = sum((1 - y_pred) * (1 - y_t), dtype=uint64)

    return (tp + tn) / (tp + tn + fp + fn)

def dice_coef(y_true, y_pred, smooth=0.0000001): #NOTE(Jesse): Computes a kind of "How similar are predictions to ground truth"
    y_t = y_true

    tp = sum(y_pred * y_t, dtype=uint64)
    fp = sum(y_pred * (1 - y_t), dtype=uint64)
    fn = sum((1 - y_pred) * y_t, dtype=uint64)

    two_tp = 2.0 * tp

    if y_true.sum() == 0: #NOTE(Jesse): Empty patch handling.
        tn = sum((1 - y_pred) * (1 - y_t), dtype=uint64)
        two_tp += tn

    return (two_tp + smooth) / (two_tp + fp + fn + smooth)

def main():
    from osgeo import gdal, ogr
    from glob import glob
    from numpy import zeros, float32, std

    gdal.UseExceptions()
    ogr.UseExceptions()

    tsp_fps = glob(test_set_spring_label_fp_glob_match)
    assert len(tsp_fps) > 0, test_set_spring_label_fp_glob_match

    tsu_fps = glob(test_set_summer_label_fp_glob_match)
    assert len(tsu_fps) > 0, test_set_spring_label_fp_glob_match

    mt_fps = glob(train_set_src_mosaic_tile_fp + "*.zip")
    assert len(mt_fps) > 0, train_set_src_mosaic_tile_fp

    results = [[None for _ in range(len(label_categories))] for _ in range(len(time_categories))] #NOTE(Jesse): Intention is results[time_idx][label_idx]

    mt_ts_groups = [None] * len(mt_fps)
    for i, mt_fp, in enumerate(mt_fps):
        ts_fps = tsp_fps
        date_str = "_spring"
        tile_name = mt_fp.split("/")[-1]
        tile_row_col = "_".join(tile_name.split("_")[2:4]) #NOTE(Jesse): aoi6_spring_199_420_aligned_pan_512_combined_trees.zip  -> 199_420
        if tile_name.startswith("arctic_Area6_6931"):
            ts_fps = tsu_fps
            date_str = "_summer"
            tile_row_col = "_".join(tile_name.split("_")[5:7]) #NOTE(Jesse): arctic_Area6_6931_GE01-QB02-WV02-WV03_PV_199_420_mosaic.tif -> 199_420

        test_set_fps = []
        tile_row_col_date_str = tile_row_col + date_str
        for tp_fp in ts_fps:
            if tile_row_col_date_str in tp_fp:
                test_set_fps.append(tp_fp)

        mt_ts_groups[i] = (mt_fp, test_set_fps)

    for m_r_fp, test_set_fps in mt_ts_groups:
        tile_fn = m_r_fp.split("/")[-1].split(".")[0]
        time_idx = 0 if "spring" in tile_fn else 1
        label_idx = 0 if "pan_512" in tile_fn else 1

        mt_ds = gdal.Open("/vsizip/" + m_r_fp + "/NN_classification.tif")
        mt_gt = mt_ds.GetGeoTransform()

        pred_hist = zeros(256 + 1, dtype=float32) #NOTE(Jesse): +1 to bin features beyond the maximum size limit
        label_hist = zeros(256 + 1, dtype=float32)

        for ts_fp in test_set_fps:
            ts_ds = gdal.Open(ts_fp)

            label_gpkg_fp = ts_fp.replace("annotation_and_boundary.tif", "test_labels.gpkg")
            ts_geo_ds = gdal.OpenEx(label_gpkg_fp)#ogr.Open(label_gpkg_fp)
            assert ts_geo_ds.GetLayerCount() == 1
            ts_geo_lyr = ts_geo_ds.GetLayer()

            ts_lyr_name = ts_geo_lyr.GetName()

            ts_gt = ts_ds.GetGeoTransform()
            ts_arr = ts_ds.GetRasterBand(1).ReadAsArray()
            ts_arr[ts_arr > 1] = 0 #NOTE(Jesse): 0 out boundary weights

            #NOTE(Jesse): Sanity check geospatial bounds / xform
            assert mt_gt[0] <= ts_gt[0], (mt_gt[0], ts_gt[0])
            assert mt_gt[3] > ts_gt[3], (mt_gt[3], ts_gt[3])

            assert abs(mt_gt[1] - ts_gt[1]) < 0.01, (mt_gt[1], ts_gt[1])
            assert abs(mt_gt[5] - ts_gt[5]) < 0.01, (mt_gt[5], ts_gt[5])

            #NOTE(Jesse): Determine the source XY read extents from the label extents
            x0 = int(((ts_gt[0] - mt_gt[0]) / mt_gt[1]) + 0.5)
            y0 = int(((ts_gt[3] - mt_gt[3]) / mt_gt[5]) + 0.5)

            mt_arr_intersect = mt_ds.GetRasterBand(1).ReadAsArray(x0, y0, ts_ds.RasterXSize, ts_ds.RasterYSize)
            mt_arr_intersect[mt_arr_intersect <= 50] = 0
            mt_arr_intersect[mt_arr_intersect > 50] = 1
            assert mt_arr_intersect.shape == ts_arr.shape

            mt_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=ts_ds.RasterXSize, ysize=ts_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
            mt_mem_ds.GetRasterBand(1).SetNoDataValue(0)
            mt_mem_ds.SetGeoTransform(ts_gt)
            mt_mem_ds.SetProjection(ts_ds.GetProjection())
            mt_mem_band = mt_mem_ds.GetRasterBand(1)
            mt_mem_band.WriteArray(mt_arr_intersect)

            raster_meters_per_texel = 0.5
            density_meters_per_texel = 100
            density_x_texel_count = int(((density_meters_per_texel - 1) + ts_ds.RasterXSize * raster_meters_per_texel) / density_meters_per_texel)
            density_y_texel_count = int(((density_meters_per_texel - 1) + ts_ds.RasterYSize * raster_meters_per_texel) / density_meters_per_texel)

            density_gxfm = array(ts_gt)
            density_gxfm[1] = density_meters_per_texel
            density_gxfm[5] = -density_meters_per_texel

            mt_cnt_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=density_x_texel_count, ysize=density_y_texel_count, bands=1, eType=gdal.GDT_Int16)
            mt_cnt_mem_ds.SetGeoTransform(density_gxfm)
            mt_cnt_mem_ds.SetProjection(ts_ds.GetProjection())

            mt_area_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=density_x_texel_count, ysize=density_y_texel_count, bands=1, eType=gdal.GDT_Float32)
            mt_area_mem_ds.SetGeoTransform(density_gxfm)
            mt_area_mem_ds.SetProjection(ts_ds.GetProjection())

            ts_cnt_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=density_x_texel_count, ysize=density_y_texel_count, bands=1, eType=gdal.GDT_Int16)
            ts_cnt_mem_ds.SetGeoTransform(density_gxfm)
            ts_cnt_mem_ds.SetProjection(ts_ds.GetProjection())

            ts_area_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=density_x_texel_count, ysize=density_y_texel_count, bands=1, eType=gdal.GDT_Float32)
            ts_area_mem_ds.SetGeoTransform(density_gxfm)
            ts_area_mem_ds.SetProjection(ts_ds.GetProjection())

            #NOTE(Jesse): Temporarily do OGR dance to get predicted gpkg features.  Should be done in stage1
            mt_v_mem_ds = gdal.GetDriverByName("Memory").Create('', 0, 0, 0, gdal.GDT_Unknown)#ogr.GetDriverByName("Memory").CreateDataSource("")
            mt_v_mem_lyr = mt_v_mem_ds.CreateLayer("predicted_labels", ts_ds.GetSpatialRef(), ogr.wkbPolygon)

            gdal.Polygonize(mt_mem_band, mt_mem_band, mt_v_mem_lyr, -1)

            if mt_v_mem_lyr.GetFeatureCount() > 0:
                pred_hist_lyr = mt_v_mem_ds.ExecuteSQL("select count(ST_Area(GEOMETRY)) as bin_count, cast(sqrt(ST_Area(GEOMETRY)) as INT) as bin_idx from predicted_labels group by bin_idx", dialect="sqlite")
                for ftr in pred_hist_lyr:
                    bin_count = ftr.GetField("bin_count")
                    bin_idx = ftr.GetField("bin_idx")
                    if bin_idx > 256:
                        bin_idx = 256
                        pred_hist[bin_idx] += bin_count
                        continue

                    pred_hist[bin_idx] = bin_count

                ro = gdal.RasterizeOptions(bands=[1], burnValues=1, SQLStatement="select ST_Centroid(GEOMETRY) from predicted_labels", SQLDialect="SQLITE", allTouched=False, add=True)
                gdal.Rasterize(mt_cnt_mem_ds, mt_v_mem_ds, options=ro)

                ro = gdal.RasterizeOptions(bands=[1], attribute="area", SQLStatement="select ST_Centroid(GEOMETRY), ST_Area(GEOMETRY) as area from predicted_labels", SQLDialect="SQLITE", allTouched=False, add=True)
                gdal.Rasterize(mt_area_mem_ds, mt_v_mem_ds, options=ro)

            if ts_geo_lyr.GetFeatureCount() > 0:
                #NOTE(Jesse): For some insane reason the name of the "geometry" column can differ in gpkg, so we discover it here.
                feat_def = ts_geo_lyr.GetLayerDefn()
                geom_field_def = feat_def.GetGeomFieldDefn(0)
                geom_column_name = geom_field_def.GetName()

                #NOTE(Jesse): To properly "escape" the geometry column name, it must be double quoted within the string, so we use single quotes to satisfy Python
                # since we can't use double quote strings inside a python string defined by double quotes.  Spent an afternoon to figure out THAT incantation.
                label_hist_lyr = ts_geo_ds.ExecuteSQL(f'select count(ST_Area("{geom_column_name}")) as bin_count, cast(sqrt(ST_Area("{geom_column_name}")) as INT) as bin_idx from "{ts_lyr_name}" group by bin_idx', dialect="sqlite")
                for ftr in label_hist_lyr:
                    bin_count = ftr.GetField("bin_count")
                    bin_idx = ftr.GetField("bin_idx")
                    if bin_idx > 256:
                        bin_idx = 256
                        label_hist[bin_idx] += bin_count
                        continue

                    label_hist[bin_idx] = bin_count

                ro = gdal.RasterizeOptions(bands=[1], burnValues=1, SQLStatement=f'select ST_Centroid({geom_column_name}) from "{ts_lyr_name}"', SQLDialect="SQLITE", allTouched=False, add=True)
                gdal.Rasterize(ts_cnt_mem_ds, ts_geo_ds, options=ro)

                ro = gdal.RasterizeOptions(bands=[1], attribute="area", SQLStatement=f'select ST_Centroid({geom_column_name}), ST_Area({geom_column_name}) as area from "{ts_lyr_name}"', SQLDialect="SQLITE", allTouched=False, add=True)
                gdal.Rasterize(ts_area_mem_ds, ts_geo_ds, options=ro)

            mt_v_mem_ds.ReleaseResultSet(pred_hist_lyr)
            ts_geo_ds.ReleaseResultSet(label_hist_lyr)

            loss = tversky(ts_arr, mt_arr_intersect)
            acc = accuracy(ts_arr, mt_arr_intersect)
            dc = dice_coef(ts_arr, mt_arr_intersect)

            cnt_diff = ts_cnt_mem_ds.GetRasterBand(1).ReadAsArray() - mt_cnt_mem_ds.GetRasterBand(1).ReadAsArray()
            area_diff = ts_area_mem_ds.GetRasterBand(1).ReadAsArray() - mt_area_mem_ds.GetRasterBand(1).ReadAsArray()

            result = {"mosaic_tile_filename": m_r_fp, "label_file_name": ts_fp, "loss": loss, "accuracy": acc, "dice_coefficient": dc, "label_histogram": label_hist, "prediction_histogram": pred_hist}
            if results[time_idx][label_idx] is None:
                results[time_idx][label_idx] = [result]
            else:
                results[time_idx][label_idx].append(result)

            mt_v_mem_lyr = None
            mt_v_mem_ds = None
            mt_arr_intersect = None

            mt_cnt_mem_ds = None
            mt_area_mem_ds = None

            ts_arr = None
            ts_ds = None

            ts_cnt_mem_ds = None
            ts_area_mem_ds = None

            mt_mem_band = None
            mt_mem_ds = None

        mt_ds = None
        mt_gt = None

    lowest_loss_time_idx = -1
    lowest_loss_label_idx = -1
    lowest_loss = 10
    losses = []
    for t_idx in range(len(time_categories)):
        for l_idx in range(len(label_categories)):
            r_list = results[t_idx][l_idx]

            r_loss = 0
            for r in r_list:
                r_loss += r["loss"]
            r_loss /= len(r_list)
            losses.append(r_loss)

            if r_loss < lowest_loss:
                lowest_loss_label_idx = l_idx
                lowest_loss_time_idx = t_idx
                lowest_loss = r_loss

    print(time_categories[lowest_loss_time_idx], label_categories[lowest_loss_label_idx], lowest_loss, std(losses))

main()
