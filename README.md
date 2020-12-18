# landmark-detection
This is a very rudimentary attempt at a classical computer vision solution to the [object co-dection problem](https://cvgl.stanford.edu/projects/codetection/index.html). The gist of the approach is to take a subset of your image set (the train set), do sift matching to each pairing between images and extract a small number of swatches. Swatches are areas of images that match to others with high confidence. These swatches can be used to match to the other images in the set (the test set).

### Extract Swatches (Train)
```
python sift_loader.py --num_swatches 4 --train_directory "./train_image_set/" --swatch_directory "./swatch_output/" --render False
```
This command shows all possible arguements with their default values. The render argument shows figures of the matches and clusters found. This will output the num_swatches most confident swatches found.
### Match Swatches (Test)
```
python match_swatches.py --test_directory "./train_image_set/" --swatch_directory "./swatch_output/" --render False
```
This command shows all possible arguements with their default values. The render argument shows figures of the matches and clusters found. This will compare swatches to test images and output confidences for each image.

### Notes
Commands can be very slow because SIFT is slow in general. There are other (probably better) approaches that are beyond the scope of this project.
Developed on `Python 3.7.6` requires `opencv` and `matplotlib`. Created for final project for [CSC292: Mobile Visual Computing](https://www.cs.rochester.edu/courses/572/fall2020/index.html) at the University of Rochester.
