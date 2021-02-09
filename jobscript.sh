#!/bin/bash --login

# setup CSF variables
#$ -cwd                               # Run the job in the current directory
#$ -pe smp.pe 5                       # 5 processes (4 cores for model, one for handler)
#$ -t 1-40                            # A job-array with 40 "tasks", numbered 1...40
#$ -m bea                             # email me on begin, error, abort
#$ -M jonathan.huck@manchester.ac.uk  # set email address

# load anaconda & environment
module load apps/anaconda3/5.2.0
source activate bayesian

# set timeout for Theano to avoid module lock problem
# https://github.com/pymc-devs/pymc3/issues/1463
# http://deeplearning.net/software/theano/library/config.html#environment-variables
export THEANO_FLAGS="compile.timeout=2000,warn.ignore_bug_before=all"

# run script
python run.py --radius 100 --resolution 20 --census './data/CommunityDefinition.shp' --survey './data/survey.csv' --surveyxy './data/survey-xy.csv' --mapme './data/mapme.shp' --gps './data/gps.shp' --clip_poly ./data/aoi/exploded/$SGE_TASK_ID.shp --id $SGE_TASK_ID
