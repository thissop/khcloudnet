def get_statistics(masks_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/annotated-batches', image_strip_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/strips', key_csv:str='/Users/tkiker/Documents/GitHub/khcloudnet/data/sampled_rows.csv'): 
    import os 
    import numpy as np 
    import pandas as pd 
    from tqdm import tqdm 
    from PIL import Image
    import cv2
    import scipy

    image_names = []
    cloud_pixel_counts = []
    areas = []

    original_cutout_length = 1

    for mask_dir in tqdm(os.listdir(masks_dir)[25:]):
        if '.' not in mask_dir:
            mask_dir = os.path.join(masks_dir, mask_dir)
            for file in os.listdir(mask_dir):
                if 'png' in file: 
                    mask_image_path = os.path.join(mask_dir, file)

                    try: 
                        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
                        original_cutout_length = mask_image.shape[0] 

                    except: 
                        print(mask_image_path)

                    non_zero_sum = np.count_nonzero(mask_image)
                    cloud_pixel_counts.append(non_zero_sum)

                    labeled_array, num_features = scipy.ndimage.label(mask_image)

                    unique_labels, counts = np.unique(labeled_array, return_counts=True)

                    if len(counts) > 1: # Ignore the background label (0)
                        areas.extend(counts[1:].tolist())  # Remove background label

                    image_names.append(file.split('.')[0])


    image_names = [i.split('-')[0]+'-'+i.split('-')[1] for i in image_names]
    image_names = list(set(image_names))
    center_lats = []
    center_longs = []

    cutouts_per_strip = 4
    covered_area_kmsqs = []
    covered_area_percentages = []
    catalogue_df = pd.read_csv(key_csv)
    for image_name in list(set(image_names)):
        catalogue_row = catalogue_df.loc[catalogue_df['Entity ID'] == image_name].squeeze()
        total_area_kmsq = catalogue_row['area_kmsq_equal_area']
        center_lats.append(catalogue_row['Center Latitude dec'])
        center_longs.append(catalogue_row['Center Longitude dec'])

        cutout_image_path = f'/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/strips/{image_name}.jpg'
        cutout_image = Image.open(cutout_image_path)
        area_factor = cutouts_per_strip*cutout_image.height/cutout_image.width

        covered_area_kmsqs.append(area_factor*total_area_kmsq)
        covered_area_percentages.append(100*area_factor)

    # mean/std covered area: 

    print('covered areas of strips: mean: ', np.mean(covered_area_percentages), "std", np.std(covered_area_percentages))

    # mean/std 
    coverage_percentages = 100*np.array(cloud_pixel_counts)/original_cutout_length**2

    print(f'Mean (%): {np.mean(coverage_percentages)}, STD (%): {np.std(coverage_percentages)}')

    # % with less than 5% cloud Coverage 

    print(f'% with Less than 5% Cloud Coverage: {100*len(np.where(coverage_percentages<5)[0])/len(coverage_percentages)}')

    # Total Coverage 
    print(f'Total coverage (km^2): {np.sum(covered_area_kmsqs)}')

    # most covered one: 

    print('One with the most coverage km^2: ', np.max(covered_area_kmsqs), image_names[np.argmax(covered_area_kmsqs)])

    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    def get_country_code(lat, lon):
        geolocator = Nominatim(user_agent="geo_locator")
        
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True)
            if location and 'country_code' in location.raw['address']:
                return location.raw['address']['country_code'].upper()
            else:
                return "N/A"
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Error with geolocation service: {e}")
            return "N/A"

    country_codes = [get_country_code(center_lats[i], center_longs[i]) for i in range(len(center_longs))]
    print('Number of Countries: '+len(list(set(country_codes)))-1)

get_statistics()

## for some reason, D3C1214-300719F004-0_3.png is causing problems (mask can't be read even though I can open it in finder??)

'''
cloud coverage of cutouts: Mean (%): 17.73680295241116, STD (%): 28.522967619583444
% with Less than 5% Cloud Coverage: 59.959623149394346
total cutout area: 873170.3939149701
One with the most coverage km^2:  118413.11325192924 D3C1203-100051F010
number of countries: 59
covered areas of strips: mean:  52.856052949477736 std 20.8979366706334
'''