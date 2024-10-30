def sendRequest(url, data, apiKey = None):  
    r"""
    Sends a POST request to the specified URL with the provided data and optional API key for authentication.

    Args:
        url (str): The API endpoint to send the request to.
        data (dict): The data to be sent in the body of the request, typically as a JSON object.
        apiKey (str, optional): API key for authenticated requests. Defaults to None.

    Returns:
        dict: The parsed JSON response from the API if successful.

    Raises:
        SystemExit: If an error occurs in the request (e.g., invalid endpoint, missing data, authentication error, or server issues).
    """
   
    import json 
    import requests 
    import sys 

    serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    
    pos = url.rfind('/') + 1
    endpoint = url[pos:]
    json_data = json.dumps(data)
    
    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {'X-Auth-Token': apiKey}              
        response = requests.post(url, json_data, headers = headers)    
    
    try:
        httpStatusCode = response.status_code 
        if response == None:
            print("No output from service")
            sys.exit()
        
        else: 
            output = json.loads(response.text)	

            if output['errorCode'] != None:
                print("Failed Request ID", output['requestId'])
                print(output['errorCode'], "-", output['errorMessage'])
                sys.exit()
            
            if httpStatusCode in [404, 401, 400]: 
                print(f"{httpStatusCode} Not Found")
                sys.exit()

    except Exception as e: 
          response.close()
          pos=serviceUrl.find('api')
          print(f"Failed to parse request {endpoint} response. Re-check the input {json_data}. The input examples can be found at {url[:pos]}api/docs/reference/#{endpoint}\n")
          sys.exit()

    response.close()    
        
    return output['data']

def download_metadata(entity_IDs:list, username:str, password:str, gpkg_name:str='usgs_metadata', dataset_name:str=None):
    r"""
    Downloads metadata for specified entities from the USGS API and saves the metadata as a GeoPackage file.

    Args:
        entity_IDs (list): A list of entity IDs for which metadata will be retrieved.
        username (str): USGS EarthExplorer username for authentication.
        password (str): USGS EarthExplorer password for authentication.
        gpkg_name (str, optional): Name of the output GeoPackage file. Defaults to 'usgs_metadata'.
        dataset_name (str, optional): The dataset name, if needed. Defaults to None.

    Returns:
        list: A list of geometries corresponding to the entities' spatial coverage.

    Raises:
        SystemExit: If metadata retrieval fails or if an entity ID is not found.
    """

    import numpy as np
    from tqdm import tqdm 
    import geopandas as gpd 
    from shapely.geometry import Polygon

    serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"

    cloud_covers = []
    start_dates  = []
    end_dates = []
    geometries = []

    for entity_ID in entity_IDs: 

        payload = {'username' : username, 'password' : password}

        apiKey = sendRequest(serviceUrl + "login", payload)

        if dataset_name is not None: 
            dataset_names = [dataset_name]

        else: 
            dataset_names = ['CORONA2', 'DECLASSII', 'DECLASSIII']

        metadata = None

        for dataset_name in dataset_names: 

            payload = {'datasetName' : dataset_name, 'entityId': entity_ID}                     
            metadata = sendRequest(serviceUrl + "scene-metadata", payload, apiKey)

            if metadata is not None: 
                cloud_cover = metadata['cloudCover'] # None
                #spatial_bounds = metadata['spatialBounds']['coordinates'][0] # {'type': 'Polygon', 'coordinates': [[[47.653, 66.129], [47.653, 66.556], [49.903, 66.556], [49.903, 66.129], [47.653, 66.129]]]}
                spatial_coverage = metadata['spatialCoverage']['coordinates'][0] # {'type': 'Polygon', 'coordinates': [[[47.653, 66.39], [49.789, 66.129], [49.903, 66.298], [47.794, 66.556], [47.653, 66.39]]]}
                temporal_coverage = metadata['temporalCoverage']

                cloud_covers.append(cloud_cover)
                start_dates.append(temporal_coverage['startDate']) #  # '1982-07-02 00:00:00-05'
                end_dates.append(temporal_coverage['endDate']) # '1982-07-02 00:00:00-05'
                geometries.append(spatial_coverage)

                break

        if metadata is None:
            print('Error: the requested entityId is not in the relevant datasets, please double check it.') 

    polygons = [Polygon(coords) for coords in geometries]
    geometries = [np.flip(i) for i in geometries][0]

    gdf = gpd.GeoDataFrame({'entityID': entity_IDs, 
                            'cloudCover': cloud_covers,
                            'temporalCoverageStart': start_dates,
                            'temporalCoverageEnd': end_dates},
                            geometry=polygons, crs="EPSG:4326")

    # Save to file

    gdf.to_file(f'{gpkg_name}.gpkg', layer='metadata', driver='GPKG')

    return geometries

def download_browse_image(image_output_dir, entity_IDs, missions, operations_numbers, cameras): 
    r"""
    Downloads browse images from USGS based on specified entity IDs, missions, operation numbers, and cameras, and saves them as JPEG files. The lists should come from your metadata file. 

    Args:
        image_output_dir (str): Directory where the downloaded images will be saved.
        entity_IDs (list): List of entity IDs for the images.
        missions (list): List of mission numbers corresponding to the images.
        operation_numbers (list): List of operation numbers corresponding to the images.
        cameras (list): List of camera codes for the respective images.

    Returns:
        None
    """

    from tqdm import tqdm 
    import os 
    import urllib.request 

    for i, j, k, l in tqdm(zip(entity_IDs, missions, operations_numbers, cameras)):
        try: 
            file_name = f'{i}.jpg' 
            image_path = os.path.join(image_output_dir, file_name) 

            if not os.path.exists(image_path):
                operation_number = ((5-len(str(k)))*'0')+str(k) 
                data_link = f'https://ims.cr.usgs.gov/browse/declass3/{j}/{operation_number}/{l}/{file_name}' 

                # https://ims.cr.usgs.gov/browse/declass3/1213-1/00123/F/D3C1213-100123F001.jpg
                #print(data_link) 

                image_path = os.path.join(image_output_dir, file_name) 
                urllib.request.urlretrieve(data_link, filename=image_path) 

        except Exception as e: # if nothing gets downloaded, check what error exception is coming up. Keep track of the ones that throw errors (some images (e.g. 1%) won't be able to be downloaded like this for some reason from experiance. 
            print(e)
            continue 

