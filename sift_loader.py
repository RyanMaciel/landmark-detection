import os
import cv2
import matplotlib.pyplot as plt
import json
import traceback
import sys
import argparse


from clustering import DBSCANClustering
from image_matching import match_images
from util import load_images, max_indices, get_image_patches_for_cluster, add_cluster_annotations

# Load a subset of the training images and compute all matching clusters
# From these extract image patches (the areas of the images that we think 
# co-occur) and return n that we are most confident in (n given num_candidate_matches). 
def extract_candidate_swatches(images, num_candidate_matches = 4, render_output = False):
    image_swatches = []
    confidences = []
    for i in range(len(images)):
        for j in range(i, len(images)):
            if i != j:
                try:
                    left_image = images[i]
                    right_image = images[j]
                    pair_match_points = match_images(left_image, right_image, render_output=render_output)
                    cluster_labels = DBSCANClustering(pair_match_points, 0.3, 4, left_image.shape, right_image.shape)

                    if len(cluster_labels) > 0:
                        if render_output and max(cluster_labels) > 0:
                            output_image = left_image.copy()
                            result_image = add_cluster_annotations(pair_match_points, cluster_labels, output_image)
                            cv2.imshow("Result", result_image);cv2.waitKey();cv2.destroyAllWindows()

                        # array of array with clustered point matches
                        clusters = []
                        for _ in range(max(cluster_labels)+1):
                            clusters.append([])
                        for k in range(len(pair_match_points)):
                            cluster_label = cluster_labels[k]
                            if cluster_label > 0:
                                clusters[cluster_label].append(pair_match_points[k])

                        # get image patches for each cluster.
                        for cluster in clusters:
                            if len(cluster) > 0:
                                image_patch_1, image_patch_2 = get_image_patches_for_cluster(cluster, left_image, right_image, margin_percentage = 0.075)
                                # cv2.imshow("Result", image_patch_1);cv2.waitKey();cv2.destroyAllWindows()
                                # cv2.imshow("Result", image_patch_2);cv2.waitKey();cv2.destroyAllWindows()
                                image_swatches.append([image_patch_1, image_patch_2])
                                print('for ' + str(len(image_swatches)-1) + ' we got confidence ' + str(len(cluster)))
                                confidences.append(len(cluster))
                except Exception as e:
                    print("SOMETHING WENT WRONG")
                    print(e)
                    traceback.print_exception(*sys.exc_info())
    
    filtered_swatches = []
    max_conf_indices = max_indices(confidences, num_candidate_matches)
    for conf_index in max_conf_indices:
        filtered_swatches.append(image_swatches[conf_index])

    return filtered_swatches


parser = argparse.ArgumentParser(
    description="Find similar patches in image set using SIFT and DBSCAN"
)
parser.add_argument(
    "--train_directory",
    default='./train_image_set/',
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
    swatch_directory = args.swatch_directory
    image_set_directory = args.train_directory

    images, _ = load_images(image_set_directory)
    image_swatches = extract_candidate_swatches(images, 3, render_output=args.render)

    # clear output folder
    for filename in os.listdir(swatch_directory):
        os.remove(swatch_directory+filename)
    for i in range(len(image_swatches)):
        cv2.imwrite(swatch_directory + 'out_' + str(2*i) + '.jpg', image_swatches[i][0])
        cv2.imwrite(swatch_directory + 'out_' + str((2*i) + 1) + '.jpg', image_swatches[i][1])

    # with open('inter_data.json', 'w') as json_file:
    #     match_points = match_images(left_image, right_image, True)
    #     json.dump(match_points, json_file)

    # with open('inter_data.json') as json_file:
    #     match_points = json.load(json_file)
    #     cluster_labels = DBSCANClustering(match_points, 200, 4)
    #     edit_image = left_image

    #     result_image = add_cluster_annotations(match_points, cluster_labels, edit_image)
    #     cv2.imshow("Result", result_image);cv2.waitKey();cv2.destroyAllWindows()

        
    #     # array of array with clustered point matches
    #     clusters = []
    #     for i in range(max(cluster_labels)+1):
    #         clusters.append([])
    #     for i in range(len(match_points)):
    #         cluster_label = cluster_labels[i]
    #         if cluster_label > 0:
    #             clusters[cluster_label].append(match_points[i])

    #     # get image patches for each cluster.
    #     for cluster in clusters:
    #         if len(cluster) > 1:
    #             image_patch_1, image_patch_2 = get_image_patches_for_cluster(cluster, left_image, right_image, margin = 100)
    #             cv2.imshow("Result", image_patch_1);cv2.waitKey();cv2.destroyAllWindows()
    #             cv2.imshow("Result", image_patch_2);cv2.waitKey();cv2.destroyAllWindows()

    


