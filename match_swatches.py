from image_matching import match_images
from util import load_images
from clustering import DBSCANClustering
swatch_directory = './swatch_output/'
image_set_directory = '../rochester_image_set'
if __name__ == "__main__":

    # assumption is that the swatches contain different
    # versions of the single object that we are interested in
    swatches = load_images(swatch_directory)
    test_images = load_images(image_set_directory, log_directory = True)

    image_confidences = []
    for test_image in test_images:
        image_sum_confidence = 0
        for swatch in swatches:
            swatch_match_points = match_images(swatch, test_image, render_output=True)
            cluster_labels = DBSCANClustering(swatch_match_points, 200, 4)
            if len(cluster_labels) > 0:
                for c in cluster_labels:
                    if c > 0:
                        image_sum_confidence += 1
        image_confidences.append(image_sum_confidence)
    print(image_confidences)
                
        