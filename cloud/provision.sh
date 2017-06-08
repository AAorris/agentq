#!/bin/bash

gcloud compute instances create $GCLOUD_VM \
  --image cos-stable-58-9334-74-0 \
  --image-project cos-cloud \
  --machine-type f1-micro
