#!/bin/bash

# Checking if both username and password are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <docker-username> <docker-password>"
    exit 1
fi

DOCKER_USERNAME=$1
DOCKER_PASSWORD=$2

# Logging in to Docker
echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" --password-stdin

# Building the Docker image
docker build -t redhunter .

# Running the Docker container and mounting the experiments directory
docker run --name redhunter -v $(pwd)/src/experiments:/Redundancy-Hunter/src/experiments redhunter

