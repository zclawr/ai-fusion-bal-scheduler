#!/bin/bash

# NOTE: THIS IS ONLY CONFIGURED FOR INPUT/OUTPUT PARSING, NOT FULL BAL RUNS CURRENTLY

# $1 : scheduler job type [output, input] (refers to data used for parsing)
# $2 : s3 path to inputs/outputs (excluding leading slash, but include trailing slash (if pointing to directory))
cp ./config/launch_template.yaml ./config/launch.yaml
JOB_TO_RUN=""

if [[ $1 == "output" ]]; then
  JOB_TO_RUN="load-and-parse-outputs"
else
  JOB_TO_RUN="load-and-parse-inputs"
fi

echo "dataset:
  default:
    hparam:
      _s3path: $2
run:
  model: [$JOB_TO_RUN]
  dataset: [default]" >> ./config/launch.yaml
echo "Configured launch.yaml, making job"
make job