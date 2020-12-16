
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