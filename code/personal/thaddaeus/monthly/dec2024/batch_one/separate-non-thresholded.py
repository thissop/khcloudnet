def separate_non_thresholded(thresholded_df, original_dir):
    import os 
    import shutil 
    import pandas as pd

    non_thresholded_dir = os.path.join(original_dir, 'not-thresholded')

    if not os.path.exists(non_thresholded_dir): 
        os.mkdir(non_thresholded_dir)

    df = pd.read_csv(thresholded_dir)
    names = list(df['Name'])

    for filename in os.listdir(original_dir): 
        if '.png' in filename: 
            thresholded = False 
            for name in names: 
                if name in filename: 
                    thresholded = True
            
            if not thresholded: 
                shutil.copyfile(os.path.join(original_dir, filename), os.path.join(non_thresholded_dir, filename))





thresholded_dir = '/Users/tkiker/Downloads/Threshold Masked Images - Sheet1.csv'
original_dir = '/Users/tkiker/Desktop/batch_77'

separate_non_thresholded(thresholded_dir, original_dir)