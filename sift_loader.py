import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

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


def lewis_kanade_approach(image1, image2):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    
    old_gray = image1
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(image1)

    frame_gray = image2
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        image2 = cv2.circle(image2,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(image2,mask)
    # cv2.imshow('frame',img)
    cv2.imshow("Result", img);cv2.waitKey();cv2.destroyAllWindows()


    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


# given two images, return a set of matching points based on
# Sift keypoints w/ FLANN matching. 
def match_images(image1, image2, render_output=False):
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(sigma=1.5)
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    
    # FLANN matching adapted from openCV tutorial:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # FLANN Matching
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    good_matches = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

            # Extraction of coordinates detailed here:
            # https://stackoverflow.com/questions/46607647/sift-feature-matching-point-coordinates
            point1 = keypoints_1[m.queryIdx].pt
            point2 = keypoints_2[m.trainIdx].pt
            good_matches.append([point1, point2])
     
            ## Draw pairs in purple, to make sure the result is ok
            cv2.circle(image1, (int(point1[0]),int(point1[1])), 10, (255,0,255), -1)
            cv2.circle(image2, (int(point2[0]),int(point2[1])), 10, (255,0,255), -1)

    
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    img3 = cv2.drawMatchesKnn(image1,keypoints_1,image2,keypoints_2,matches,None,**draw_params)

    #plt.imshow(img3,),plt.show()
    if render_output:
        cv2.imshow("Result", img3);cv2.waitKey();cv2.destroyAllWindows()
    return good_matches

# Get the neighbors of a match (given as point_index, the index of the match)
# in point_matches by measuring the distances of the first points (corresponding
# to the "left" image) plus the second points (corresponding to the "right" image)
def get_neighbors(point_matches, distance_limit, point_index):
    neighbors = []
    start_point = point_matches[point_index]
    for match_index in range(len(point_matches)):
        match = point_matches[match_index]
        # The distance between points in the original image (left)
        left_distance = ( (match[0][0] - start_point[0][0]) ** 2 + (match[0][1] - start_point[0][1]) ** 2 ) ** (1/2)
        # distance between their neighbors
        right_distance = ((match[1][0] - start_point[1][0]) ** 2 + (match[1][1] - start_point[1][1]) ** 2) ** (1/2)
        if left_distance + right_distance < distance_limit:
            neighbors.append(match_index)
    return neighbors
        

# Do a unique type of DBSCAN clustering. See get_neighbors as
# to what makes it interesting
def DBSCANClustering(point_matches, distance_limit, min_points):

    # labels[i] will be the cluster label of point_match index i.
    # 0 will be an unlabeled point. -1 will be noise
    labels = []
    for _ in range(len(point_matches)):
        labels.append(0)

    current_cluster = 0
    for i in range(len(point_matches)):
        point_match = point_matches[i]
        if labels[i] != 0:
            continue
        neighbors = get_neighbors(point_matches, distance_limit, i)
        if len(neighbors) < min_points: 
            labels[i] = -1
            continue
        
        current_cluster += 1
        labels[i] = current_cluster
        neighbors.remove(i)

        current_neighbor = 0
        while current_neighbor < len(neighbors):
            q_match_index = neighbors[current_neighbor]
            if labels[q_match_index] == -1:
                labels[q_match_index] == current_cluster
            if labels[q_match_index] != 0:
                current_neighbor +=1
                continue
            labels[q_match_index] = current_cluster

            neighbors_to_add = get_neighbors(point_matches, distance_limit, q_match_index)
            if len(neighbors_to_add) >= min_points:

                # add the new neighbors that are not already in neighbors.
                for n in neighbors_to_add:
                    if n not in neighbors:
                        neighbors.append(n)
            current_neighbor +=1

    return labels
        

# Given a clustered set of matches, get an image patch contained within
# the cluster. We can then check this against the other images.
def get_image_patches_for_cluster(cluster_matches, image_1, image_2, margin = 0):

    # get bounds for left image
    left_min_x = int(min(cluster_matches, key=lambda k:k[0][0])[0][0])
    left_max_x = int(max(cluster_matches, key=lambda k:k[0][0])[0][0])
    left_min_y = int(min(cluster_matches, key=lambda k:k[0][1])[0][1])
    left_max_y = int(max(cluster_matches, key=lambda k:k[0][1])[0][1])
    # factor in margin
    left_min_x = max(left_min_x-margin, 0)
    left_max_x = min(left_max_x+margin, image_1.shape[1])
    left_min_y = max(left_min_y-margin, 0)
    left_max_y = min(left_max_y+margin, image_1.shape[0])
    #opencv does height then width. (took me some insanely annoying debugging to discover)
    cropped_image_1 = image_1[left_min_y:left_max_y, left_min_x:left_max_x]

    # get bounds for right image
    right_min_x = int(min(cluster_matches, key=lambda k:k[1][0])[1][0])
    right_max_x = int(max(cluster_matches, key=lambda k:k[1][0])[1][0])
    right_min_y = int(min(cluster_matches, key=lambda k:k[1][1])[1][1])
    right_max_y = int(max(cluster_matches, key=lambda k:k[1][1])[1][1])

    # factor in margin
    right_min_x = max(right_min_x-margin, 0)
    right_max_x = min(right_max_x+margin, image_2.shape[1])
    right_min_y = max(right_min_y-margin, 0)
    right_max_y = min(right_max_y+margin, image_2.shape[0])

    cropped_image_2 = image_2[right_min_y:right_max_y, right_min_x:right_max_x]

    return (cropped_image_1, cropped_image_2)


def add_cluster_annotations(match_points, cluster_labels, image):
    num_classes = max(cluster_labels)
    jump = 255/num_classes
    for i in range(len(match_points)):
        cluster_num = cluster_labels[i]
        cv2.circle(image, (int(match_points[i][0][0]),int(match_points[i][0][1])), 7, (jump*cluster_num,jump*cluster_num,255-(jump*cluster_num)), -1)
    return image

def extract_candidate_swatches():
    images = load_images()
    image_swatches = []
    print(len(images))
    for i in range(len(images)):
        for j in range(i, len(images)):
            if i != j:
                try:
                    print(i)
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
                            if len(cluster) > 1:
                                image_patch_1, image_patch_2 = get_image_patches_for_cluster(cluster, left_image, right_image, margin = 100)
                                # cv2.imshow("Result", image_patch_1);cv2.waitKey();cv2.destroyAllWindows()
                                # cv2.imshow("Result", image_patch_2);cv2.waitKey();cv2.destroyAllWindows()
                                image_swatches += [image_patch_1, image_patch_2]
                except Exception as e:
                    print("SOMETHING WENT WRONG")
                    print(e)
    return image_swatches



if __name__ == "__main__":
    image_swatches = extract_candidate_swatches()
    for i in range(len(image_swatches)):
        cv2.imwrite('./swatch_output/out_' + str(i) + '.jpg', image_swatches[i])

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

    


