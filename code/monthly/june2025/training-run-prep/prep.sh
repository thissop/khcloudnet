mkdir -p train_images_10 train_masks_10 test_images_10 test_masks_10

# Sample 10% of training data
find train_images/train_images -maxdepth 1 -type f | shuf -n 1915 | while read fname; do
    cp "$fname" train_images_10/
    cp "train_masks/train_masks/$(basename "$fname")" train_masks_10/
done

# Sample 10% of test data
find test_images/test_images -maxdepth 1 -type f | shuf -n 200 | while read fname; do
    cp "$fname" test_images_10/
    cp "test_masks/test_masks/$(basename "$fname")" test_masks_10/
done

rm -rf __MACOSX

mv train_images/train_images/* train_images/
mv train_masks/train_masks/* train_masks/
mv test_images/test_images/* test_images/
mv test_masks/test_masks/* test_masks/

# Count files in 10% training subset
echo "Train Images 10%:" && find /discover/nobackup/tkiker/data/khcloudnet/train_images_10 -type f | wc -l
echo "Train Masks 10%:" && find /discover/nobackup/tkiker/data/khcloudnet/train_masks_10 -type f | wc -l

# Count files in 10% test subset
echo "Test Images 10%:" && find /discover/nobackup/tkiker/data/khcloudnet/test_images_10 -type f | wc -l
echo "Test Masks 10%:" && find /discover/nobackup/tkiker/data/khcloudnet/test_masks_10 -type f | wc -l


#### make sure all files have matches (if nothing gets printed then we good)
cd /discover/nobackup/tkiker/data/khcloudnet
comm -3 <(ls train_images_10 | sort) <(ls train_masks_10 | sort)
comm -3 <(ls test_images_10 | sort) <(ls test_masks_10 | sort)

#rm -r train_images/train_images train_masks/train_masks test_images/test_images test_masks/test_masks

### get ready for jesse format 
cd /discover/nobackup/tkiker/data/khcloudnet

echo "Processing training set..."
for img in train_images_10/*.png; do
    base=$(basename "$img" .png)
    cp "train_masks_10/${base}.png" "train_images_10/${base}_annotation_and_boundary.png"
done

echo "Processing test set..."
for img in test_images_10/*.png; do
    base=$(basename "$img" .png)
    cp "test_masks_10/${base}.png" "test_images_10/${base}_annotation_and_boundary.png"
done

echo "All files processed and renamed successfully."

