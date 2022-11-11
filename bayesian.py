"""
Bayesian MCC model based upon Dirichlet Distribution for Community Belonging in
 North Belfast

Calculates weighted 'observation' scores from various evidence sources and uses
 them to predict probability of observing members of that group in that location
 (as a proxy for community belonging)

References:
	https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/Ch3_IntroMCMC_PyMC3.ipynb
	https://github.com/WillKoehrsen/probabilistic-programming/blob/master/Estimating%20Probabilities%20with%20Bayesian%20Inference.ipynb
	https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815
	http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
	https://gist.github.com/tboggs/8778945
	---
	https://eigenfoo.xyz/bayesian-modelling-cookbook/#fixing-divergences
	https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html
	https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
	http://www.ericmjl.com/notebooks/dirichlet-multinomial-bayesian-proportions/
	---
	https://blogs.oracle.com/datascience/introduction-to-bayesian-inference
	http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/
	https://www.analyticsvidhya.com/blog/2016/06/bayesian-statistics-beginners-simple-english/
	https://nbviewer.jupyter.org/github/AllenDowney/ThinkBayes2/blob/master/solutions/dirichlet_soln.ipynb
	https://towardsdatascience.com/markov-chain-monte-carlo-in-python-44f7e609be98
	https://github.com/WillKoehrsen/ai-projects/blob/master/markov_chain_monte_carlo/markov_chain_monte_carlo.ipynb
	https://nbviewer.jupyter.org/github/AllenDowney/ThinkBayes2/blob/master/solutions/dirichlet_soln.ipynb

Conda Env:
	conda install -c conda-forge geopandas rasterio pymc3 mkl-service

@author jonnyhuck
"""

from math import ceil
from os import makedirs
from os.path import exists
from arviz import plot_trace
from shapely.geometry import Point
from logging import getLogger, ERROR
from rasterio import open as rio_open
from numpy import array, zeros, ones, maximum
from rasterio.transform import xy, from_origin
from pymc3 import Model, Dirichlet, Multinomial, sample, summary


def array2Coords(transform, row, col):
	"""
	* convert between array position and coords
	*  params are row,col (y,x) as expected by rasterio
	*  returns coords at the CENTRE of the cell
	"""
	return xy(transform, row, col)


def getMapmeGroups(precise_matches):
	"""
	Calculate weighted counts for each group from Map-Me data
	 Zoom is -12, so 13 (lowest) = 1 vote, 18 (highest) = 6 votes
	"""
	# adjust zoom to a score and weight by idw2
	precise_matches['weighted'] = precise_matches['idw2'] * (precise_matches['zoom']-12)

	# get counts, weighted by zoom level
	catholics = precise_matches[precise_matches.Community == 1]['weighted'].sum()
	protestants = precise_matches[precise_matches.Community == 2]['weighted'].sum()
	mixed = precise_matches[precise_matches.Community == 3]['weighted'].sum()

	# return the counts
	return catholics, protestants, mixed


def getGpsGroups(precise_matches):
	'''
	Calculate weighted counts for each group from GPS data
	'''

	# get counts catholics, weighted by number of users that produced them
	grouped = precise_matches[precise_matches.comm == 1].groupby('id_user')['idw2'].sum()
	catholics = grouped.sum() * len(grouped)

	# get counts protestants, weighted by number of users that produced them
	grouped = precise_matches[precise_matches.comm == 2].groupby('id_user')['idw2'].sum()
	protestants = grouped.sum() * len(grouped)

	# get counts mixed, weighted by number of users that produced them (uses other as a proxy for mixed)
	grouped = precise_matches[precise_matches.comm == 3].groupby('id_user')['idw2'].sum()
	mixed = grouped.sum() * len(grouped)

	# return the counts
	return catholics, protestants, mixed


def getSurveyGroups(precise_matches):
	'''
	Calculate weighted counts for each group
	 Simply weighted linearly with distance
	'''

	# counts are simply IDW2
	catholics = precise_matches[precise_matches.Community == "Catholic"]['idw2'].sum()
	protestants = precise_matches[precise_matches.Community == "Protestant"]['idw2'].sum()

	# return the counts (with 0 for mixed as not represented in the data)
	return catholics, protestants, 0


def f(mask, aoi_id, plot=False):
	'''
	* run parallel process
	'''

	# turn off pymc3 logging
	getLogger("pymc3").setLevel(ERROR)

	# get transform object for the dataset (nw corner & resolution)
	transform = from_origin(mask['bounds'][0], mask['bounds'][3], mask['resolution'], mask['resolution'])

	# check that output directory is there
	if not exists("./out/"):
		makedirs("./out/")

	# seed data and uncertainty arrays for the study area and build dictionary to control outputs
	c_data = zeros((
        ceil((mask['bounds'][3] - mask['bounds'][1]) / mask['resolution']),
		ceil((mask['bounds'][2] - mask['bounds'][0]) / mask['resolution'])
        ))
	outputs = {
		'catholic':     {'path': f'./out/{aoi_id}_catholic.tif',  'mean': c_data,         'low': c_data.copy(), 'high': c_data.copy() },
		'protestant':   {'path': f'./out/{aoi_id}_protestant.tif','mean': c_data.copy(),  'low': c_data.copy(), 'high': c_data.copy() },
		'mixed':        {'path': f'./out/{aoi_id}_mixed.tif',     'mean': c_data.copy(),  'low': c_data.copy(), 'high': c_data.copy() }
		}

	# extract list of group names
	groups = array(list(outputs.keys()))

	# use try-finally so if it fails we can see where it got up to
	# try:

	print(f"AOI Dimensions: {c_data.shape[1]}x{c_data.shape[0]}px")

	# loop through rows and columns in the dataset
	for row in range(c_data.shape[0]):
		for col in range(c_data.shape[1]):

			print(f"\t...{row * c_data.shape[1] + col} of {c_data.shape[0] * c_data.shape[1]} ({(row * c_data.shape[1] + col)/(c_data.shape[0] * c_data.shape[1])*100:.2f}%)")

			# get coordinates for the point
			point = Point(array2Coords(transform, row, col))

			''' calculate hyperparameters (priors) '''

			# get the census data for the census Small Area that contains the point
			possible_matches = mask['census'].iloc[list(mask['census'].sindex.intersection(point.bounds))]
			district = possible_matches.loc[possible_matches.contains(point)][['pcCatholic', 'pcProtesta', 'pc_Other', 'pc_None']]

			# make sure that there was a match at all!
			if len(district.index) > 0:

				# compute proportions for the three groups
				# replace zeros for 1s as you are not allowed 0's in the hyperparameters (gives Bad initial energy error)
				alphas = maximum(ones(3),array([
					int(round(district['pcCatholic'].iloc[0])),
					int(round(district['pcProtesta'].iloc[0])),
					int(round(district['pc_Other'].iloc[0] + district['pc_None'].iloc[0]))
					]))

			else:
				# if no matches, have equal belief for each group
				alphas = array([1, 1, 1])

			''' calculate observations '''

			# init lists for observations
			c = []
			n = []

			# construct the radius for analysis
			polygon = point.buffer(mask['radius'])

			# loop through each dataset
			for i, gdf in mask['datasets'].items():

				# check that there is data available (this is if no data has been
				#  passed in the mask as the clip polygon does not intersect any)
				if len(gdf.index) > 0:

					# get data points within and get IDW2 multiplier
					possible_matches = gdf.iloc[list(gdf.sindex.intersection(polygon.bounds))]
					observations = possible_matches.loc[possible_matches.within(polygon)]

					observations['idw2'] = (1 - observations.geometry.distance(point) / mask['radius']) ** 2

					# check that there is data available (this is if data has been
					#  passed but the buffer polygon does not intersect it)
					if len(observations) > 0:

						# get weighted group counts for the current dataset
						if i == 'mapme':
							catholics, protestants, mixed = getMapmeGroups(observations)
						elif i == 'gps':
							catholics, protestants, mixed = getGpsGroups(observations)
						elif i == 'survey':
							catholics, protestants, mixed = getSurveyGroups(observations)

						# index and int the scores for each dataset
						sums = [catholics, protestants, mixed]

						# catch error caused by no probabilities
						if sum(sums) > 0:

							print(sums)

							# process into correct format
							sums = [int(round(i / sum(sums) * 100)) for i in sums]

							# append to observations list
							c.append(sums)
							n.append(sum(sums))

							# TODO: DO I WANT ALL THESE 0'S OR ARE THEY GOING TO CAUSE PROBLEMS?

						else:
							# if not matches, just append some empty data
							c.append([0, 0, 0])
							n.append(0)
					else:
						# if not matches, just append some empty data
						c.append([0, 0, 0])
						n.append(0)
				else:
					# if not matches, just append some empty data
					c.append([0, 0, 0])
					n.append(0)

			# convert observations np array
			c = array(c)
			n = array(n)

			# print(alphas, c, n)
			# print()

			''' run model '''

			# start making MCC model
			with Model() as model:

				# TODO: LOOK INTO TESTVALS FOR PARAMETERS
				# https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/Ch3_IntroMCMC_PyMC3.ipynb#Intelligent-starting-values

				# parameters of the Multinomial are from a Dirichlet
				parameters = Dirichlet('parameters', a=alphas, shape=3)

				# observed data is from a Multinomial distribution
				observed_data = Multinomial('observed_data', n=n, p=parameters, shape=3, observed=c)

				with model:

					# estimate the Maximum a Posterior
					# start = find_MAP()	#don't use this - it prevents convergence!

					# sample from the posterior (NUTS is default so is not explicitly stated)
					trace = sample(
						# start=start,                # start at the MAP to increase chance of convergence -- DON'T DO THIS!
						draws=1000,                 # number of sample draws
						chains=4,                   # number of chains in which the above are drawn (match cores)
						cores=1,                    # max permitted by library
						tune=500,                   # how many will be discarded (>=50% of draws)
						discard_tuned_samples=True, # discard the tuning samples
						progressbar = False,        # avoid unnecessarilly filling up the output file
						target_accept = 0.9			# up from 0.8 to avoid false positives: https://eigenfoo.xyz/bayesian-modelling-cookbook/#fixing-divergences
						)

					if plot:
						plot_trace(trace, show=True)

					# retrieve summary data
					results = summary(trace)
					results.index = groups

			# output the result to the datasets
			for k, v in outputs.items():
				v['mean'][row, col] = results.loc[k, 'mean']
				v['low'][row, col] = results.loc[k, 'hpd_3%']
				v['high'][row, col] = results.loc[k, 'hpd_97%']

	# if we get an error - print some debugging info
	# except Exception as e:
	#     print("\n--- EXCEPTION ---")
	#     print(e)
	#     print(row, col, point)
	#     if (sums):
	#         print(sums)
	#     else:
	#         print("sums not defined yet")
	#     print(c, n)
	#
	# # whatever happens, output the results to files
	# finally:

	# loop through outputs
	for g in outputs.values():

		# output dataset to raster (hardcoded crs as was causing error)
		with rio_open(g['path'], 'w', driver='GTiff', height=g['mean'].shape[0],
			width=g['mean'].shape[1], count=3, dtype='float64', crs="EPSG:29902",
			transform=transform
		) as out:

			# add data and uncertainties as raster bands
			out.write(g['mean'], 1)
			out.write(g['low'], 2)
			out.write(g['high'], 3)


### this is just for testing ###
if __name__ == '__main__':

	print("This file should only be run directly for testing purposes. For production, use run.py")

	# get missing imports
	import theano
	import pandas as pd
	from pathlib import Path
	from pandas import read_csv
	from warnings import simplefilter
	from geopandas import GeoDataFrame, read_file
	from pickle import dump, load, HIGHEST_PROTOCOL

	# resolve theano problem by ignoring internal warning on C++ compiler
	theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
	pd.options.mode.chained_assignment = None
	simplefilter(action='ignore', category=FutureWarning)

	# set args
	resolution = 20
	radius = 100
	aoi = [int(x) for x in Point(334005, 376245).buffer(resolution/2).bounds]
	aoi_pad = [int(x) for x in Point(334005, 376245).buffer(radius).bounds]

	print("loading data...")

	# parses and analyses data if needed, otherwise will just load stored data
	# to force update, just delete data file
	if not Path('./pickle-jar/mask.pkl').is_file():
		print("No pickle file found, loading data from scratch (takes a long time).")

		# load data
		mapme = read_file('./data/mapme.shp')
		mapme.sindex
		gps = read_file('./data/gps.shp')
		gps.sindex
		census = read_file('./data/CommunityDefinition.shp')
		census.sindex
		survey = read_csv('./data/survey.csv').join(read_csv('./data/survey-xy.csv').set_index('participant'),
			on='Part.No', how='inner').loc[:,['Part.No', 'Community', 'x', 'y']]
		survey = GeoDataFrame(survey, geometry=[Point(xy) for xy in zip(survey.x, survey.y)])
		survey.crs = gps.crs    # just steal CRS from one of the other layers
		survey.sindex

		# construct false mask
		mask = {
			'radius': radius,
			'resolution': resolution,
			'bounds': aoi,
			'census': census.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
			'datasets': {
				'mapme': mapme.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
				'gps': gps.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
				'survey': survey.cx[aoi_pad[0]:aoi_pad[2]+1, aoi_pad[1]:aoi_pad[3]+1],
				}
			}

		# pickle the results so we don't need to go through the whole analysis every time
		with open('pickle-jar/mask.pkl', 'wb') as output:
			dump(mask, output, HIGHEST_PROTOCOL)

	# if the file does exist, can avid analysis and load previous results
	else:
		print("Pickle file found successfully, loading data.")

		# extract traces from pickle file
		with open('pickle-jar/mask.pkl', 'rb') as input:
			mask = load(input)

	print("running analysis...")

	# run the function
	f(mask, 0, True)
	print("done!")
