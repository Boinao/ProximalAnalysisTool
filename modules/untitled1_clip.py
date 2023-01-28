# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:22:17 2020

@author: Trainee
"""

import gdal
import os
#'''Input and output paths'''

inputPath = r"F:\christi\mangrv\picha_timeseries\2007/"
outputPath = r"F:\christi\mangrv\picha_timeseries\STACK\2007/"

#Input Raster and Vector Paths
bandList = [band for band in os.listdir(inputPath) if ( band[-9:-5]=="band" and band[-4:]==".tif")]
bandList


# Shapefile of Area of Interest'''
shp_clip = r"F:/christi/mangrv/picha_timeseries/ROI/PICHAVARAM.shp"

#Clip all the selected raster files with the Warp option from GDAL'''

for band in bandList:
    print(outputPath + band[:-4]+'_clip'+band[-4:])
    options = gdal.WarpOptions(cutlineDSName=shp_clip,cropToCutline=True)
    outBand = gdal.Warp(srcDSOrSrcDSTab=inputPath + band,
                        destNameOrDestDS=outputPath + band[:-4]+'_clip'+band[-4:],
                        options=options)
outBand = None


