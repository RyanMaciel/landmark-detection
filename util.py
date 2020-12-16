import os
import cv2
import sys

# https://stackoverflow.com/a/30230738
def load_images_from_folder(folder, log_directory):
    images = []
    for filename in os.listdir(folder):
        if log_directory:
            print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def load_images(directory, log_directory=False):
    return load_images_from_folder(directory, log_directory)

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

def add_cluster_annotations(match_points, cluster_labels, image):
    num_classes = max(cluster_labels)
    jump = 255/num_classes
    for i in range(len(match_points)):
        cluster_num = cluster_labels[i]
        cv2.circle(image, (int(match_points[i][0][0]),int(match_points[i][0][1])), 7, (jump*cluster_num,jump*cluster_num,255-(jump*cluster_num)), -1)
    return image

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