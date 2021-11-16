"""
graph.py
-----------

Utility functions related to graph cut and grab cut from scratch.

The retaed functions and variable naming conventions come from the
following papers

1. Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. 2004. "GrabCut": 
interactive foreground extraction using iterated graph cuts. In ACM 
SIGGRAPH 2004 Papers (SIGGRAPH '04), Joe Marks (Ed.). ACM, New York, 
NY, USA, 309-314. DOI: https://doi.org/10.1145/1186562.1015720 

2. Y. Boykov, O. Veksler and R. Zabih, "Fast approximate energy minimization 
via graph cuts," in IEEE Transactions on Pattern Analysis and Machine 
Intelligence, vol. 23, no. 11, pp. 1222-1239, Nov. 2001
"""


import numpy as np
# import pymaxflow
import maxflow
# modified to pip installed maxflow
# check http://pmneila.github.io/PyMaxflow/maxflow.html
import cv2
from past.builtins import xrange


def compute_beta_vectorized(z):
     """
     Compute the expectation beta given the gradients in an  image 

     Parameters
     ----------
     z:  (m,n) uint8 
       The input grayscale image

     Returns
     -------
     beta: float
       The expectation beta for the image as described in the original grabcut paper 
      
     """

     accumulator = 0
     m = z.shape[0]
     n = z.shape[1]

     vert_shifted = z - np.roll(z, 1, axis=0)
     temp = np.sum(np.multiply(vert_shifted, vert_shifted), axis=2)
     accumulator = np.sum(temp[1:,:])

     horiz_shifted = z - np.roll(z, 1, axis=1)
     temp = np.sum(np.multiply(horiz_shifted, horiz_shifted), axis=2)
     accumulator += np.sum(temp[:,1:])

     num_comparisons = float(2*(m*n) - m - n)
   
     beta = 1.0/(2*(accumulator/num_comparisons))

     return beta

def create_graph(img):

    """
    Creat a graph (nodes without weight) from an image 

    Parameters
    ----------
    img:  (m,n) uint8 
     The input grayscale image

    Returns
    -------
    g: pymaxflow.PyGraph object
     A graph instance whoes nodes have not yet been assigned an weight 
      
    """

    num_neighbors = 8

    num_nodes = img.shape[0]*img.shape[1] + 2
    num_edges = img.shape[0]*img.shape[1]*num_neighbors

    # g = pymaxflow.PyGraph(num_nodes, num_edges)
    g = maxflow.Graph[int](num_nodes, num_edges)
    # Creating nodes
    # g.add_node(num_nodes-2)
    nodes = g.add_nodes(num_nodes-2)

    # return g
    return nodes,g




def compute_smoothness_vectorized(z, neighborhood='eight'):
    """
    Given an image, and an optional neighborhood number, computes the pairwise 
    weights between defined neigboring pixels 

    Parameters
    ----------
    z : (m,n) uint8 
     The input grayscale image
    neighborhood : string
     If == eight, eight energies with 8 neighbours are computed,
     Else, the number of neighborhood is defaulted at four

    Returns
    -------
    energies: (n, ) float
     Pairwise energies as a list
 
      
    """

    FOUR_NEIGHBORHOOD = [(-1,0), (1,0), (0,-1), (0,1)]
    EIGHT_NEIGHBORHOOD = [(-1,0),(+1,0),(0,-1),(0,+1),(-1,-1),(-1,+1),(+1,+1),(+1,-1)]

    if neighborhood == 'eight':
        NEIGHBORHOOD = EIGHT_NEIGHBORHOOD
    else:
        NEIGHBORHOOD = FOUR_NEIGHBORHOOD

    height, width, _ = z.shape
    smoothness_matrix = dict()

    beta = compute_beta_vectorized(z)

    # (i,j) gives norm(z[i,j] - z[i-1,j])
    vert_shifted_up = z - np.roll(z, 1, axis=0)
    # (i,j) gives norm(z[i,j] - z[i+1,j])
    vert_shifted_down = z - np.roll(z, -1, axis=0) 
    # (i,j) gives norm(z[i,j] - z[i,j-1])
    horiz_shifted_left = z - np.roll(z, 1, axis=1)
    # (i,j) gives norm(z[i,j] - z[i,j+1])
    horiz_shifted_right = z - np.roll(z, -1, axis=1) 

    energies = []
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(vert_shifted_up,
                                                          vert_shifted_up), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(vert_shifted_down,
                                                          vert_shifted_down), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(horiz_shifted_left,
                                                          horiz_shifted_left), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(horiz_shifted_right,
                                                          horiz_shifted_right), axis=2)))

    # Diagnonal components
    if neighborhood == 'eight':
        # (i,j) gives norm(z[i,j] - z[i-1,j-1])
        nw = z - np.roll(np.roll(z, 1, axis=0), 1, axis=1)
        # (i,j) gives norm(z[i,j] - z[i-1,j+1])
        ne = z - np.roll(np.roll(z, 1, axis=0), -1, axis=1)
        # (i,j) gives norm(z[i,j] - z[i+1,j+1])
        se = z - np.roll(np.roll(z, -1, axis=0), -1, axis=1)
        # (i,j) gives norm(z[i,j] - z[i+1,j-1])
        sw = z - np.roll(np.roll(z, -1, axis=0), 1, axis=1) 

        energies.append(np.exp(-1 * beta * np.sum(np.multiply(nw, nw), axis=2)))
        energies.append(np.exp(-1 * beta * np.sum(np.multiply(ne, ne), axis=2)))
        energies.append(np.exp(-1 * beta * np.sum(np.multiply(se, se), axis=2)))
        energies.append(np.exp(-1 * beta * np.sum(np.multiply(sw, sw), axis=2)))
   
    return energies


def get_unary_energy_vectorized(alpha, k, gmms, pixels):
    """
    Given a list of pixels and a gmm (gmms contains both the foreground and 
    the background GMM, but alpha helps us pick the correct one), returns the 
    -log(prob) of each belonging to the component specified by k.

    Parameters
    ----------
    alpha : int 
     Integer specifying background or foreground (0 or 1)
    k : (n, ) int
     Array with each element corresponding to which component the corresponding 
     pixel in the pixels array belongs to
    gmms : an utils.GMM instance
     Gaussian mixture models for the foreground or the background as indexed by
     alpha
    pixels : (n,3) uint.8
      Array of pixels

    Returns
    -------
    energies: (n, ) float
     Unary energies(foreground or background as specified by alpha) of the given 
     pixels 
 
      
    """


    pi_base = gmms[alpha].weights
    pi = pi_base[k].reshape(pixels.shape[0])
    pi[pi==0] = 1e-15

    dets_base = np.array([gmms[alpha].gaussians[i].sigma_det for i in xrange(len(gmms[alpha].gaussians))])
    dets = dets_base[k].reshape(pixels.shape[0])
    dets[dets==0] = 1e-15

    means_base = np.array([gmms[alpha].gaussians[i].mean for i in xrange(len(gmms[alpha].gaussians))])
    means = np.swapaxes(means_base[k], 1, 2)
    means = means.reshape((means.shape[0:2]))

    cov_base = np.array([gmms[alpha].gaussians[i].sigma_inv for i in xrange(len(gmms[alpha].gaussians))])
    cov = np.swapaxes(cov_base[k], 1, 3)
    cov = cov.reshape((cov.shape[0:3]))

    term = pixels - means
    middle_matrix = np.array([np.sum(np.multiply(term, cov[:, :, 0]),axis=1),
                              np.sum(np.multiply(term, cov[:, :, 1]),axis=1),
                              np.sum(np.multiply(term, cov[:, :, 2]),axis=1)]).T

    # Not really the log_prob, but a part of it
    log_prob = np.sum(np.multiply(middle_matrix, term), axis=1)

    return -np.log(pi*1.0/np.sqrt(dets)*np.exp(-0.5*log_prob))

def round_int(x):
    ALOT = 1e6
    vals = max(min(x, ALOT), -ALOT)
    # if x == float("inf"):
    #     return
    # elif x == float("-inf"):
    #     return float('nan') # or x or return whatever makes sense
    # return int(round(x))
    return round(vals)

def add_edge_vectorized(graph,i,j,cap,rev_cap):
    assert i.size == j.size
    assert i.size == cap.size
    assert i.size == rev_cap.size
    for l in range(i.size):
        graph.add_edge(i[l], j[l], cap[l], rev_cap[l])
    return graph

# perform the relaxed graph cut
def graphcut(img, alpha,rect,foreground_gmm,background_gmm,flow_vector, gamma,
             num_iterations=1, num_components=5):

    """
    Given an image, return the partition of the image using customized graphcut 
    as described in the in hand object scanning paper

    Parameters
    ----------
    img:  (m,n,3) uint8 
     The input color image
    alpha : int 
     Integer specifying background or foreground (0 or 1)
    rect : (4, ) float
     A list of 4 elements of (x,y,w,h), which is the zoomed in bounding box of 
     the segment from the previous frame, this is to reduce the amount of 
     computation needed to perform graphcut (Graphcut only computed within the
     region of this rect
    foreground_gmm : an utils.GMM instance
      Gaussian mixture models for the foreground
    background_gmm : an utils.GMM instance
      Gaussian mixture models for the background 
    flow_vector : (m,n) float
      A flow gradient that describe the amount of shifts per pixel between 
      frames
    gamma : int
      A ballancing constant used according to the original graphcut paper

    Returns
    -------
    result: (m,n ) uint8
     A partition of the image where 255 indicates foreground and 0 indicaes background
 
      
    """



   
    img2=img.copy()

    # trim down the cut area to the projected rect estimated from the last frame
    img=img[int(rect[0]):int(rect[0]+rect[2]),int(rect[1]):int(rect[1]+rect[3])]
    alpha=alpha[int(rect[0]):int(rect[0]+rect[2]),int(rect[1]):int(rect[1]+rect[3])]
    flow_vector=flow_vector[int(rect[0]):int(rect[0]+rect[2]),int(rect[1]):int(rect[1]+rect[3])]

    # compute the pairwise smoothness term for both color and optical flow image
    # and choose the smaller of the two 
    user_definite_background = np.where(alpha==0)
    pairwise_energies_1 = compute_smoothness_vectorized(img, neighborhood='eight')
    pairwise_energies_2 = compute_smoothness_vectorized(flow_vector, neighborhood='eight')
    pairwise_energies= np.fmin(pairwise_energies_1, pairwise_energies_2)

    # Get GMM components based on color from the given fgd,bgf model
    pixels = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
    foreground_components = foreground_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))
    background_components = background_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))

    # Compute Unary energies for each node (foreground and background energies)
    # graph = create_graph(img)
    node_ids, graph = create_graph(img)
    theta = (background_gmm, foreground_gmm, None, None)
    foreground_energies = get_unary_energy_vectorized(1, foreground_components.reshape((img.shape[0]*img.shape[1], 1))
                                                      , theta, pixels)
    background_energies = get_unary_energy_vectorized(0, background_components.reshape((img.shape[0]*img.shape[1], 1))
                                                      , theta, pixels)
    energy_differences = np.subtract(foreground_energies,background_energies)
    energy_min = np.fmin(foreground_energies, background_energies)
    better_count = np.where(energy_differences<10)
    foreground_energies[better_count] = energy_min[better_count]

    # Assign Unary energy for user defined background
    # Large foreground energy: gamma*9 as used in opencv implementation
    # Small background energy: 0
    foreground_energies = foreground_energies.reshape(alpha.shape)
    background_energies = background_energies.reshape(alpha.shape)
    
    foreground_energies[user_definite_background] = gamma*9
    background_energies[user_definite_background] = 0

    # update graph with the energies
    for h in xrange(img.shape[0]):
        for w in xrange(img.shape[1]):
            index = h*img.shape[1] + w
            # void add_tweights(node_id i, tcaptype cap_source, tcaptype cap_sink);
            # graph.add_tweights(index, foreground_energies[h][w], background_energies[h][w])
            # try:
            #     graph.add_tedge(index, foreground_energies[h][w], background_energies[h][w])
            # except:
            #     print(index, foreground_energies[h][w], background_energies[h][w])
            graph.add_tedge(index, foreground_energies[h][w], round_int(background_energies[h][w]))

    # Compute pairwise weights
    NEIGHBORHOOD = [(-1,0),(+1,0),(0,-1),(0,+1),(-1,-1),(-1,+1),(+1,+1),(+1,-1)]
    src_h = np.tile(np.arange(img.shape[0]).reshape(img.shape[0], 1), (1, img.shape[1]))
    src_w = np.tile(np.arange(img.shape[1]).reshape(1, img.shape[1]), (img.shape[0], 1))
    src_h = src_h.astype(np.int32)
    src_w = src_w.astype(np.int32)

    for i, energy in enumerate(pairwise_energies):
        if i in [1,3,6,7]:
            continue
        height_offset, width_offset = NEIGHBORHOOD[i]

        dst_h = src_h + height_offset
        dst_w = src_w + width_offset

        idx = np.logical_and(np.logical_and(dst_h >= 0, dst_h < img.shape[0]),
                            np.logical_and(dst_w >= 0, dst_w < img.shape[1]))

        src_idx = src_h * img.shape[1] + src_w
        dst_idx = dst_h * img.shape[1] + dst_w

        src_idx = src_idx[idx].flatten()
        dst_idx = dst_idx[idx].flatten()
        weights = energy.astype(np.float32)[idx].flatten()
        weights = gamma*weights

        # graph.add_edge_vectorized(src_idx, dst_idx, weights, weights)
        # graph.add_edge(src_idx, dst_idx, weights, weights)
        graph = add_edge_vectorized(graph, src_idx, dst_idx, weights, weights)
    # perform mincut/maxflow with pymaxflow
    graph.maxflow()
    # partition = graph.what_segment_vectorized()
    partition = graph.get_grid_segments(node_ids)

    partition = partition.reshape(alpha.shape)
    blank=np.zeros(img2.shape[:2])
    blank[int(rect[0]):int(rect[0]+rect[2]),int(rect[1]):int(rect[1]+rect[3]) ]=partition*255
    result = blank.astype(dtype=np.uint8)
 
    return result
