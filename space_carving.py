#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import scipy.io
import numpy as np
import cv2
import glob
import open3d as o3d
import mcubes

# Load camera matrices
data = scipy.io.loadmat("data/dino_Ps.mat")
data = data["P"]
projections = [data[0, i] for i in range(data.shape[1])]

# load images
files = sorted(glob.glob("data/*.ppm"))
images = []
for f in files:
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(float)
    im /= 255
    images.append(im[:, :, ::-1])
    

# get silouhette from images
imgH, imgW, __ = images[0].shape
silhouette = []
for im in images:
    temp = np.abs(im - [0.0, 0.0, 0.75])
    temp = np.sum(temp, axis=2)
    y, x = np.where(temp <= 1.1)
    im[y, x, :] = [0.0, 0.0, 0.0]
    im = im[:, :, 0]
    im[im > 0] = 1.0
    im = im.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    silhouette.append(im)

    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    
#%%
# create voxel grid
s = 120
x, y, z = np.mgrid[:s, :s, :s]
pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
pts = pts.T
nb_points_init = pts.shape[0]
xmax, ymax, zmax = np.max(pts, axis=0)
pts[:, 0] /= xmax
pts[:, 1] /= ymax
pts[:, 2] /= zmax
center = pts.mean(axis=0)
pts -= center
pts /= 5
pts[:, 2] -= 0.62

pts = np.vstack((pts.T, np.ones((1, nb_points_init))))

filled = []
for P, im in zip(projections, silhouette):
    uvs = P @ pts
    uvs /= uvs[2, :]
    uvs = np.round(uvs).astype(int)
    x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < imgW)
    y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < imgH)
    good = np.logical_and(x_good, y_good)
    indices = np.where(good)[0]
    fill = np.zeros(uvs.shape[1])
    sub_uvs = uvs[:2, indices]
    res = im[sub_uvs[1, :], sub_uvs[0, :]]
    fill[indices] = res 
    
    filled.append(fill)

filled = np.vstack(filled)

# the occupancy is computed as the number of camera in which the point "seems" not empty
occupancy = np.sum(filled, axis=0)

# Marching cubes
occ = occupancy.reshape((120, 120, 120))
vertices, triangles = mcubes.marching_cubes(occ, 30)

# Visualisation with open3d
mesh = o3d.geometry.TriangleMesh()
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])