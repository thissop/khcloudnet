if __name__ == "__main__":
    exit()

class unet_config:
    def default_config(self):
        #NOTE(Jesse): Version 1 is the "old" version, which lacks attention layers in the upsampling path (and trained on 256x256x2 patches).  
        #             Version 2 has attention layers
        #             Version 3 has deep supervision on the final upsample path

        self.UNet_version = 3

        #NOTE(Jesse): How many patches are collected at once for inference or training (use the highest number available on your system, usually a power of 2)
        self.batch_size = 96

        #NOTE(Jesse): Underscores represent private fields and should only be accessed via get* procedures defined at the bottom ONLY UNLESS these are not being set by command line switches.

        #NOTE(Jesse): name and __order__ of the raster band as specified in the geotiff file
        #self.raster_bands = ("pan", "ndvi")

        self.raster_bands = ("p", "b", "g", "r", "n", "v")
        self.raster_band_indices = (1, 2, 3, 4, 5, 6)
        self.label_bands = ("label", "weight")

        self.is_deterministic = False
        self.is_multi_gpu = False

        #NOTE(Jesse): Width * Height * NumberChannels
        self.batch_input_shape = [256, 256, len(self.raster_bands)] #NOTE(Jesse): Dimensions of training raster data provided to the UNET
        self.batch_label_shape = [self.batch_input_shape[0], self.batch_input_shape[1], 1] #NOTE(Jesse): Dimensions of label raster data provided to the UNET

        self.raster_shape = [512, 512, len(self.raster_bands)] #NOTE(Jesse): Dimensions of raster data loaded from disk
        self.label_shape = [self.raster_shape[0], self.raster_shape[1], 1] #NOTE(Jesse): Dimensions of raster label data loaded from disk

        self.shape_xy = self.batch_input_shape[0]
        self.step_xy = int(self.shape_xy - 32) #NOTE(Jesse): 16m^2  processing overlap.  Assumes 0.5 meters per pixel resolution.

    def set_raster_bands(self, rb):
        self.raster_bands = rb
        self.batch_input_shape[2] = len(rb)
        self.raster_shape[2] = len(rb)

        #NOTE(Jesse): Pan and Pan NDVI mosaic configuration
        if len(self.raster_bands) == 1: #NOTE(Jesse): Pan
            self.raster_band_indices = (1,)
            return

        elif len(self.raster_bands) == 2: #NOTE(Jesse): Pan NDVI
            self.raster_band_indices = (1, 2)
            return

        indices = []
        wide_band_ordering = ("p", "b", "g", "r", "n", "v")
        for i, rb_v in enumerate(rb):
            for w_i, wbo in enumerate(wide_band_ordering[i:]):
                if wbo == rb_v:
                    indices.append(i + w_i + 1) 
                    break

            assert rb_v in wide_band_ordering, rb_v

        self.raster_band_indices = tuple(indices)

    def __init__(self):
        self.default_config()