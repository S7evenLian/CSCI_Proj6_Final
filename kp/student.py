# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
from random import sample

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    #[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    # Your code here #
    ##################

    # construct the uv_matrix
    uv_matrix = np.reshape(Points_2D,(Points_2D.shape[0]*Points_2D.shape[1],1))

    ### construct the A matrix ###
    # the total number of rows
    m = Points_2D.shape[0]*2

    # col 0:2
    col_0_to_2 = np.zeros((m,3))
    col_0_to_2[::2] = Points_3D

    # col 3
    col_3 = np.zeros((m,1))
    col_3[::2] = 1

    # col 4:6
    col_4_to_6 = np.roll(col_0_to_2,1,axis=0)

    # col 7
    col_7 = np.roll(col_3,1,axis=0)

    # col 8:10
    col_8_to_10 = col_0_to_2 + col_4_to_6
    col_8_to_10 = -col_8_to_10 * uv_matrix

    # assemble A
    A = np.hstack([col_0_to_2,col_3,col_4_to_6,col_7,col_8_to_10])

    # apply the least square regression
    M, residuals, rank, s  = np.linalg.lstsq(A,uv_matrix)

    # append M34
    M = np.append(M,1)

    # reshape the final M
    M = np.reshape(M,(3,4))

    return M

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    ##################

    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    Q = M[:,0:3]
    M4 = M[:,3]
    
    Center = np.dot(-np.linalg.inv(Q),M4)

    return Center

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix
def estimate_fundamental_matrix(Points_a,Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    ##################
    # Your code here #
    ##################

    # Apply coordinate normalization
    u = Points_a[:,0]
    v = Points_a[:,1]
    u_p = Points_b[:,0]
    v_p = Points_b[:,1]

    cu = np.mean(u)
    cv = np.mean(v)
    cu_p = np.mean(u_p)
    cv_p = np.mean(v_p)

    cu_d = u - cu
    cv_d = v - cv
    cu_p_d = u_p - cu_p
    cv_p_d = v_p - cv_p

    s = np.sqrt(2)/np.sqrt(np.sum(np.square(cu_d)+np.square(cv_d))/Points_a.shape[0])
    s_p = np.sqrt(2)/np.sqrt(np.sum(np.square(cu_p_d)+np.square(cv_p_d))/Points_a.shape[0])

    Ta = np.array([[s,0,0],[0,s,0],[0,0,1]]) @ np.array([[1,0,-cu],[0,1,-cv],[0,0,1]])
    Tb = np.array([[s_p,0,0],[0,s_p,0],[0,0,1]]) @ np.array([[1,0,-cu_p],[0,1,-cv_p],[0,0,1]])

    Points_a = np.transpose(Ta @ np.transpose(np.hstack((Points_a,np.ones((Points_a.shape[0],1))))))
    Points_b = np.transpose(Tb @ np.transpose(np.hstack((Points_b,np.ones((Points_b.shape[0],1))))))

    Points_a = Points_a[:,0:2]
    Points_b = Points_b[:,0:2]

    # construct maxtix A
    u = Points_a[:,0]
    v = Points_a[:,1]
    u_p = Points_b[:,0]
    v_p = Points_b[:,1]
    A = np.zeros((Points_a.shape[0], 9))
    A[:,0] = u * u_p
    A[:,1] = u * v_p
    A[:,2] = u
    A[:,3] = v * u_p
    A[:,4] = v * v_p
    A[:,5] = v
    A[:,6] = u_p
    A[:,7] = v_p
    A[:,8] = 1

    # Solve a system of homogeneous linear equations
    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1,:]
    F = np.transpose(np.reshape(F, (3,3)))

    # Resolve det(F)=0 constraint using SVD
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diagflat(S) @ Vh

    # retrive the fundamental matrix
    F = np.transpose(Tb) @ F @ Ta

    return F

# Takes h, w to handle boundary conditions
def apply_positional_noise(points, h, w, interval=3, ratio=0.2):
    """ 
    The goal of this function to randomly perturbe the percentage of points given 
    by ratio. This can be done by using numpy functions. Essentially, the given 
    ratio of points should have some number from [-interval, interval] added to
    the point. Make sure to account for the points not going over the image 
    boundary by using np.clip and the (h,w) of the image. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.clip

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] ( note that it is <x,y> )
            - desc: points for the image in an array
        h :: int 
            - desc: height of the image - for clipping the points between 0, h
        w :: int 
            - desc: width of the image - for clipping the points between 0, h
        interval :: int 
            - desc: this should be the range from which you decide how much to
            tweak each point. i.e if interval = 3, you should sample from [-3,3]
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will have some number from 
            [-interval, interval] added to the point. 
    """
    ##################
    # Your code here #
    ##################

    # get the total number of the points
    n_point = points.shape[0]

    # randomly select index based on the ratio
    n_rand = int(n_point * ratio)
    ind_rand = sample(range(n_point), n_rand)

    # generate the noise based on the interval
    noise = np.random.randint(-interval, interval, size=(n_rand,2))

    # add noise to the points
    for i in range(n_rand):
        points[ind_rand[i],:] = points[ind_rand[i],:] + noise[i,:]

    # clip the boundary
    points[:,0] = np.clip(points[:,0], 0, w)
    points[:,1] = np.clip(points[:,1], 0, h)

    return points

# Apply noise to the matches. 
def apply_matching_noise(points, ratio=0.2):
    """ 
    The goal of this function to randomly shuffle the percentage of points given 
    by ratio. This can be done by using numpy functions. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.random.shuffle  

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] 
            - desc: points for the image in an array
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will be randomly shuffled.
    """
    ##################
    # Your code here #
    ##################

    # get the total number of the points
    n_point = points.shape[0]

    # randomly select index based on the ratio
    n_rand = int(n_point * ratio)
    ind_rand = sample(range(n_point), n_rand)

    # shuffle the selected index
    ind_shuffle = np.copy(ind_rand)
    np.random.shuffle(ind_rand)

    # make a copy of the original points
    points_copy = np.copy(points)

    # update with the shuffled index
    for i in range(n_rand):
        points[ind_rand[i],:] = points_copy[ind_shuffle[i],:]

    return points


# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.
    ##################
    # Your code here #
    ##################

    # initialize
    Best_Fmatrix = np.zeros((3,3))
    inliers_a = np.empty(0)
    inliers_b = np.empty(0)
    inliers_count_max = 0

    # RANSAC parameters
    ransac_run = 2000
    sample_num = 30
    n_point = matches_a.shape[0]
    threshold = 0.001

    for i in range(ransac_run):

        # temporary array for inliers
        inliers_a_tmp = np.array([])
        inliers_b_tmp = np.array([])

        # to count total number of inliers for each RANSAC run
        inliers_count = 0

        # randomly select the interest points pairs
        ind_rand = sample(range(n_point), sample_num)
        matches_a_sample = matches_a[ind_rand,:]
        matches_b_sample = matches_b[ind_rand,:]

        # estimate the fundamental matrix
        F = estimate_fundamental_matrix(matches_a_sample, matches_b_sample)

        # calculate the distance for all data pairs
        for j in range(n_point):
            x   = np.ones((3,1))
            x_p = np.ones((3,1))
            x[0]   = np.transpose(matches_a[j,0])
            x[1]   = np.transpose(matches_a[j,1])
            x_p[0] = np.transpose(matches_b[j,0])
            x_p[1] = np.transpose(matches_b[j,1])

            dist = np.transpose(x_p) @ F @ x
            # print('---')
            # print(dist)

            if abs(dist) < threshold:
                # add to the tmp inliers array
                inliers_a_tmp = np.hstack((inliers_a_tmp, matches_a[j,:]))
                inliers_b_tmp = np.hstack((inliers_b_tmp, matches_b[j,:]))

                # increment the inliers count
                inliers_count = inliers_count + 1

        # if this is a better estimate, update the output
        if inliers_count > inliers_count_max:       
            Best_Fmatrix = F
            inliers_count_max = inliers_count
            inliers_a = np.reshape(inliers_a_tmp,(int(len(inliers_a_tmp)/2),2))
            inliers_b = np.reshape(inliers_b_tmp,(int(len(inliers_b_tmp)/2),2))

    print('original number of matched points')
    print(matches_a.shape[0])
    print('number after RANSAC run')
    print(inliers_count_max)

    return Best_Fmatrix, inliers_a, inliers_b
