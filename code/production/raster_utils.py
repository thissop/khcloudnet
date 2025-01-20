def connected_component_analysis(bitmap, cutout_size:int=1000, naive:bool=True, min_max:list=[5, 400]):
    r'''
    A fast function for estimating the number of features predicted by our UNet neural network based on its output bitmap. 

    Parameters
    ----------

    > bitmap: 2D square numpy bitmap (read from tif via rasterio)
    > cutout_size: length/width of the square mini cutouts that this algorithm splits the main bitmap into for faster processing. 
    > naive: whether or not we should exclude clusters that are too small or too large. 
    > min_max: if naive is false, this array provides minimum and maximum values for canidate clusters. 
    
    Notes
    -----
    
    > Connected Components Algorithm: scans images to identify and label contiguous regions (components) of pixels that share similar properties (like intensity) and are connected to each other. 
    > 1/19/2025: I haven't worked on this code since June of 2024, and I copied it over from a project I did for an alternative to polygonization, and I grabbed some code from it for another thing in khcloudnet, so not sure if this works 100% in general cases now. 
    
    '''
    import cv2
    import numpy as np

    # Check Bitmap and cutout size
    height, width = bitmap.shape
    if height != width:
        raise Exception("Bitmap must be a square.")
    
    if height % cutout_size != 0:
        raise Exception("Bitmap dimensions must be divisible by cutout_size.")
    
    # Reshape and transpose to split into square sub-arrays of cutout_size
    num_subarrays = height // cutout_size
    subarrays = bitmap.reshape(num_subarrays, cutout_size, num_subarrays, cutout_size).transpose(0, 2, 1, 3).reshape(-1, cutout_size, cutout_size)
    
    num_labels = []
    image_labels = []
    
    # Apply thresholding and label connected components for each sub-array. Non-naive approach removes abnormally small/big "features."
    for subarray in subarrays:
        _, im_bw = cv2.threshold(subarray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        if naive: 
            n_labels, labels_im = cv2.connectedComponents(im_bw)
            num_labels.append(n_labels)
            image_labels.append(labels_im)

        else:    
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(im_bw)

            for i in range(1, n_labels):  
                area = stats[i, cv2.CC_STAT_AREA] 
                if  area < min_max[0] or area > min_max[1]:
                    labels[labels == i] = 0  
            
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != 0]
            adjusted_n_labels = len(unique_labels)
            
            num_labels.append(adjusted_n_labels)
            image_labels.append(labels)   

    num_labels = np.array(num_labels)
    image_labels = np.stack(image_labels)

    ## Reshape image_labels back to the original bitmap size: image_label_map = image_labels.reshape(num_subarrays, num_subarrays, cutout_size, cutout_size).transpose(0, 2, 1, 3).reshape(height, width)
    
    return num_labels, image_labels, subarrays#, image_label_map