import argparse
import cv2
from image_matching import match_images
from util import load_images, add_cluster_annotations
from clustering import DBSCANClustering

swatch_directory = './swatch_output/'
image_set_directory = './test_set'

def match_swatches(swatches, test_images, render_output=False):
    image_confidences = []
    for i in range(len(test_images)):
        test_image = test_images[i]
        image_sum_confidence = 0
        for swatch in swatches:
            swatch_match_points = match_images(swatch, test_image, render_output=render_output, ratio=0.8, flann_checks=120)
            
            cluster_labels = DBSCANClustering(swatch_match_points, 0.18, 6, swatch.shape, test_image.shape)
            if len(cluster_labels) > 0:
                if render_output and max(cluster_labels) > 0:
                        output_image = test_image.copy()
                        result_image = add_cluster_annotations(swatch_match_points, cluster_labels, output_image, side=1)
                        cv2.imshow("Result", result_image);cv2.waitKey();cv2.destroyAllWindows()

                for c in cluster_labels:
                    if c > 0:
                        image_sum_confidence += 1
        image_confidences.append(image_sum_confidence)
    return image_confidences



parser = argparse.ArgumentParser(
    description="Find similar patches in image set using SIFT and DBSCAN"
)
parser.add_argument(
    "--test_directory",
    default='./test_image_set/',
    type=str,
    help="Where to get images to train on (look for frequent co-occurances)",
)
parser.add_argument(
    "--swatch_directory",
    default='./swatch_output/',
    type=str,
    help="Where to output image swatches that we think co-occur frequently.",
)
parser.add_argument(
    "--render",
    "--r",
    default=False,
    type=bool,
    help="if we should render progress",
)

args = parser.parse_args()


if __name__ == "__main__":

    # assumption is that the swatches contain different
    # versions of the single object that we are interested in
    swatches, _ = load_images(args.swatch_directory)
    test_images, image_names = load_images(args.test_directory)

    result_confidences = match_swatches(swatches, test_images, render_output = args.render)
    
    print('Swatch Match Results:')
    for i in range(len(result_confidences)):
        print('=====================================')
        print('Image: ' + image_names[i])
        print('Confidence: ' + str(result_confidences[i]))
        print('=====================================')
    
                
        