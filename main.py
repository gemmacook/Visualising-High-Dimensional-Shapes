#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:25:59 2018

@author: cameron
"""
import pointCloud.PointCloud

filePath = '/home/cameron/Dropbox/T2_Dataset/molGeom/T2_2_num_molGeom.cif'
pc = PointCloud(filePath)
pc.plot()