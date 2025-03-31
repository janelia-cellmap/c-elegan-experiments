# Persistence Processing

This directory contains scripts used to generate final Zarr files. The processing involves selecting a model, deciding on postprocessing steps, and running a blockwise processor script to process the entire volume using multiple cluster jobs. Typically, the computation (inference + postprocessing) takes less than 30 minutes.

## Directory Structure

Each subdirectory includes:
- A Python script required for processing.
- A `submit.sh` script to submit the job to the cluster.
