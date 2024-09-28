#!/bin/bash

# Function to show usage
usage() {
    echo "Usage: $0 [--new-image] <docker-username> <docker-password>"
    exit 1
}

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    usage
fi

# Parse flags
NEW_IMAGE_FLAG=false
if [ "$1" == "--new-image" ]; then
    NEW_IMAGE_FLAG=true
    shift
fi

# Check if username and password are provided
if [ "$#" -ne 2 ]; then
    usage
fi

DOCKER_USERNAME=$1
DOCKER_PASSWORD=$2
IMAGE_NAME="redhunter"

# Prune unused containers and images
echo "Cleaning up unused containers and images..."
docker system prune -f

# Generate a unique container name with timestamp
CONTAINER_NAME="redhunter_$(date +%s)"

# Logging in to Docker
echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" --password-stdin

# Handle --new-image flag
if [ "$NEW_IMAGE_FLAG" == true ]; then
    # Check if the Docker image exists locally before attempting to remove it
    if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" != "" ]]; then
        echo "Removing old Docker image $IMAGE_NAME."
        docker rmi -f $IMAGE_NAME || echo "No such image: $IMAGE_NAME"
    else
        echo "No existing image $IMAGE_NAME found to remove."
    fi
fi

# Check if the Docker image exists locally
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Image $IMAGE_NAME not found. Building the Docker image."
    docker build -t $IMAGE_NAME . || { echo "Docker build failed. Check Dockerfile and network connection."; exit 1; }
else
    echo "Image $IMAGE_NAME already exists. Skipping build."
fi

# Verify the image exists after build
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Error: Docker image $IMAGE_NAME not found after build."
    exit 1
fi

# Check if a container with the same name already exists
EXISTING_CONTAINER=$(docker ps -aq -f name=$CONTAINER_NAME)

if [ -n "$EXISTING_CONTAINER" ]; then
    # Check if the existing container is running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Container $CONTAINER_NAME is already running. Starting it."
        docker start $EXISTING_CONTAINER
    else
        echo "Container $CONTAINER_NAME exists but is not running. Removing it."
        docker rm $EXISTING_CONTAINER
        echo "Creating and running a new container."
        docker run -it --name $CONTAINER_NAME -v $(pwd)/src/experiments:/Redundancy-Hunter/src/experiments --gpus all $IMAGE_NAME || { echo "Failed to run container with image $IMAGE_NAME"; exit 1; }
    fi
else
    echo "No existing container found. Creating and running a new one."
    docker run -it --name $CONTAINER_NAME -v $(pwd)/src/experiments:/Redundancy-Hunter/src/experiments --gpus all -m 32g $IMAGE_NAME || { echo "Failed to run container with image $IMAGE_NAME"; exit 1; }
fi
