#!/usr/bin/env bash
source 00_config.sh

docker build -f Dockerfile.cupy_cuda11 --tag dikorsch/cupy-cuda110 .

docker compose build $@ && docker compose config test_cupy
docker compose run test_cupy && echo "Installation Ready"
