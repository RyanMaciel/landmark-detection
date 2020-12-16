import os
import cv2
import matplotlib.pyplot as plt
import json
import traceback
import sys

from clustering import DBSCANClustering
from image_matching import match_images

# https://stackoverflow.com/a/30230738
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def load_images():
    return load_images_from_folder('../rochester_image_set')

# Given a clustered set of matches, get an image patch contained within
# the cluster. We can then check this against the other images.
# margin is the area around the bounding box that we include. It is 
# given as percentage of min(img_height, img_width)
def get_image_patches_for_cluster(cluster_matches, image_1, image_2, margin_percentage = 0):

    # left margin
    left_margin = int(margin_percentage * min(image_1.shape[0], image_1.shape[1]))
    # get bounds for left image
    left_min_x = int(min(cluster_matches, key=lambda k:k[0][0])[0][0])
    left_max_x = int(max(cluster_matches, key=lambda k:k[0][0])[0][0])
    left_min_y = int(min(cluster_matches, key=lambda k:k[0][1])[0][1])
    left_max_y = int(max(cluster_matches, key=lambda k:k[0][1])[0][1])
    # factor in margin
    left_min_x = max(left_min_x-left_margin, 0)
    left_max_x = min(left_max_x+left_margin, image_1.shape[1])
    left_min_y = max(left_min_y-left_margin, 0)
    left_max_y = min(left_max_y+left_margin, image_1.shape[0])
    #opencv does height then width. (took me some insanely annoying debugging to discover)
    cropped_image_1 = image_1[left_min_y:left_max_y, left_min_x:left_max_x]

    # right margin
    right_margin = int(margin_percentage * min(image_1.shape[0], image_1.shape[1]))
    # get bounds for right image
    right_min_x = int(min(cluster_matches, key=lambda k:k[1][0])[1][0])
    right_max_x = int(max(cluster_matches, key=lambda k:k[1][0])[1][0])
    right_min_y = int(min(cluster_matches, key=lambda k:k[1][1])[1][1])
    right_max_y = int(max(cluster_matches, key=lambda k:k[1][1])[1][1])

    # factor in margin
    right_min_x = max(right_min_x-right_margin, 0)
    right_max_x = min(right_max_x+right_margin, image_2.shape[1])
    right_min_y = max(right_min_y-right_margin, 0)
    right_max_y = min(right_max_y+right_margin, image_2.shape[0])

    cropped_image_2 = image_2[right_min_y:right_max_y, right_min_x:right_max_x]

    return (cropped_image_1, cropped_image_2)


def add_cluster_annotations(match_points, cluster_labels, image):
    num_classes = max(cluster_labels)
    jump = 255/num_classes
    for i in range(len(match_points)):
        cluster_num = cluster_labels[i]
        cv2.circle(image, (int(match_points[i][0][0]),int(match_points[i][0][1])), 7, (jump*cluster_num,jump*cluster_num,255-(jump*cluster_num)), -1)
    return image

# given an array of integers get the max n indices
def max_indices(arr, num_indices):
    max_indices = []
    for i in range(len(arr)):
        if len(max_indices) < num_indices:
            max_indices.append(i)
        else:
            current_num = arr[i]

            # find min value in max_indices array
            min_index=0
            min_value = sys.maxsize
            for j in range(len(max_indices)):
                check_index = max_indices[j]
                if arr[check_index] < min_value:
                    min_value = arr[check_index]
                    min_index = j
            
            if min_value < current_num:
                max_indices[min_index] = i
    return max_indices

# Load a subset of the training images and compute all matching clusters
# From these extract image patches (the areas of the images that we think 
# co-occur) and return n that we are most confident in (n given num_candidate_matches). 
def extract_candidate_swatches(images, num_candidate_matches = 4):
    image_swatches = []
    confidences = []
    print(len(images))
    for i in range(len(images)):
        for j in range(i, len(images)):
            if i != j:
                try:
                    left_image = images[i]
                    right_image = images[j]
                    pair_match_points = match_images(left_image, right_image)
                    cluster_labels = DBSCANClustering(pair_match_points, 200, 4)
                    if len(cluster_labels) > 0:
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
    
    print(confidences)
    filtered_swatches = []
    max_conf_indices = max_indices(confidences, num_candidate_matches)
    for conf_index in max_conf_indices:
        filtered_swatches.append(image_swatches[conf_index])

    return filtered_swatches


swatch_directory = './swatch_output/'
if __name__ == "__main__":
    images = load_images()
    image_swatches = extract_candidate_swatches(images, 3)

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

    


