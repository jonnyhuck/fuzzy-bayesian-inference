"""
* Divide and Conquour script to divide, process in parallel and merge again without having
*	to worry about edge effects.
* This version is intended to be used with call bayesian.py
*
* python run.py --radius 60 --resolution 20 --census './data/CommunityDefinition.shp' --survey './data/survey.csv' --surveyxy './data/survey-xy.csv' --mapme './data/mapme.shp' --gps './data/gps.shp' --clip_poly './data/aoi/exploded/1.shp'  --out './outputs'
"""

import theano
import pandas as pd
from bayesian import f
from math import ceil, floor
from geopandas import read_file
from warnings import simplefilter
from argparse import ArgumentParser
from pandas import DataFrame, read_csv
from geopandas import GeoDataFrame, read_file
from shapely.geometry import Point, Polygon, mapping


def roundToCeil(x, base):
	"""
	* Round up to nearest base
	"""
	return ceil(base * round(float(x) / base))


def roundToFloor(x, base):
	"""
	* Round down to nearest base
	"""
	return floor(base * round(float(x) / base))

# get settings from args
parser = ArgumentParser(description="Belfast Bayesian Belonging Tool")
parser.add_argument('--radius', help='Search Radius for evidence arond a given point', required=True)
parser.add_argument('--resolution', help='Resolution of the output raster', required=True)
parser.add_argument('--census', help='Path to census data', required=True)
parser.add_argument('--survey', help='Path to survey data', required=True)
parser.add_argument('--surveyxy', help='Path to xy data for survey', required=True)
parser.add_argument('--mapme', help='Path to map-me data', required=True)
parser.add_argument('--gps', help='Path to GPS data', required=True)
parser.add_argument('--clip_poly', help='The area of interest for this analysis (permits parallel processing)', required=True)
parser.add_argument('--id', help='ID for the AOI', required=True)
args = vars(parser.parse_args())

# get args
radius = int(args['radius'])
resolution = int(args['resolution'])
census_path = args['census']
survey_path = args['survey']
surveyxy_path = args['surveyxy']
mapme_path = args['mapme']
gps_path = args['gps']
clip_path = args['clip_poly']
aoi_id = args['id']

# resolve theano problem by ignoring internal warning on C++ compiler
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
pd.options.mode.chained_assignment = None
simplefilter(action='ignore', category=FutureWarning)

print("reading in data...")

# get geodataframes
mapme = read_file(mapme_path)
mapme.sindex
gps = read_file(gps_path)
gps.sindex
census = read_file(census_path)
census.sindex

# get clip polygon
clip_poly = read_file(clip_path).geometry.iloc[0]

# load and inner join survey data then make geodataframe
survey = read_csv(survey_path).join(read_csv(surveyxy_path).set_index('participant'),
	on='Part.No', how='inner').loc[:,['Part.No', 'Community', 'x', 'y']]
survey = GeoDataFrame(survey, geometry=[Point(xy) for xy in zip(survey.x, survey.y)])
survey.crs = gps.crs    # just steal CRS from one of the other layers
survey.sindex

# dictionary of datasets
datasets = {'mapme': mapme, 'gps': gps, 'survey': survey}

print("preparing data...")

'''
* This block just outputs the desired output bounds for the dataset
'''
# get bounds for study area (intersect of bounds for each input layer)
# allBounds = list(zip(mapme.total_bounds, gps.total_bounds, survey.total_bounds))
# total_bounds = [max(allBounds[0]), max(allBounds[1]), min(allBounds[2]), min(allBounds[3])]
# bounds = [
# 	roundToFloor(total_bounds[0], resolution),  # minx (west)
# 	roundToFloor(total_bounds[1], resolution),  # miny (south)
# 	roundToCeil(total_bounds[2], resolution),   # maxx (east)
# 	roundToCeil(total_bounds[3], resolution)    # maxy (north)
# 	]
#
# from geopandas import GeoSeries
# GeoSeries(
# 	Polygon.from_bounds(
# 		roundToFloor(bounds[0], resolution),
# 		roundToFloor(bounds[1], resolution),
# 		roundToCeil(bounds[2], resolution),
# 		roundToCeil(bounds[3], resolution)
# 		),
# 	crs="EPSG:29902").to_file("./data/aoi.shp")
# exit()

# get bounds as int
bounds = [int(x) for x in clip_poly.bounds]

# get padded clipping polygon to allow for the radius
aoi_pad = [bounds[0] - radius,	bounds[1] - radius, bounds[2] + radius, bounds[3] + radius]

# construct mask with necessary data
mask = {
	'radius': radius,
	'resolution': resolution,
	'bounds': bounds,
	'census': census.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
	'datasets': {
		'mapme': mapme.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
		'gps': gps.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
		'survey': survey.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
		},
	}

print("running analysis...")

# run the bayesian analysis on the current clip area
results = f(mask, aoi_id)
# results are written in f

# print how long it took
print("done!")
