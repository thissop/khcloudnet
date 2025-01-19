import os
from tqdm import tqdm 

def remove_duplicates_by_name(directory):
    seen_files = set()
    duplicate_count = 0

    for root, _, files in tqdm(os.walk(directory)):
        for file in files:
            if file in seen_files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                duplicate_count += 1
            else:
                seen_files.add(file)

    return duplicate_count

directory_path = "/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/cutouts"

duplicates_removed = remove_duplicates_by_name(directory_path)
print(f"Total duplicate files removed: {duplicates_removed}")
