import cv2
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