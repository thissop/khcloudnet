1. Make miniconda enviroment in terminal 
```python 
conda create --name satellite310 python=3.10
# maybe it will ask you to type y and click enter 
conda activate satellite310

pip install pandas, matplotlib, numpy, smplotlib


```

save version of declassiii-catalog.csv file that only has rows where month of the observation is "06" "07" or "08" and the Camera Resolution == "2 to 4 feet"

Then for like ten lines get the needed info to download images into a directory with my `download_browse_image()` function in `code/production/usgs_utils.py`

once you've done that, make a histogram of the observation years of all the data with matplotlib and all the months 

next task: sample the images according to the readme.md or email. 