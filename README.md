# Fuzzy Bayesian Inference
This repository contains code in support of an academic manuscript currently in review

## Abstract
The problem of mapping regions with socially derived boundaries has been a topic of discussion in the GIS literature for many years. Fuzzy and probabilistic approaches have frequently been suggested as solutions, but none have been adopted, likely because of the difficulties associated with determining suitable membership functions, which are often as arbitrary as the crisp boundaries that they seek to replace. This research therefore presents Fuzzy Bayesian Inference, which is a novel approach to fuzzy geographical modelling that replaces the membership function with a possibility distribution that is estimated using Bayesian inference. In this method, data from multiple sources are combined to construct a distribution that estimates both the degree to which a given location is a member of a given set and the level of uncertainty associated with that estimate. This new approach is demonstrated through a case study in which census data is combined with perceptual and behavioural evidence to model the territory of two segregated groups (Catholics and Protestants) in Belfast, Northern Ireland, UK. 

## Usage
The main FBI code is contained in `bayesian.py`, which is called from `run.py` (which handles CLI args etc.) - both are well commented. This version was run on a HPC facility at the University of Manchester using the commands contained in `jobscript.sh`.

## Data
The data from the publication are located [here](/data). Unfortunately, the GPS and questionnaire data cannot be shared publicly due to our ethical obligations to the participants, as they inherently reveal identifiable information about individual participants and could place them at significant risk.
