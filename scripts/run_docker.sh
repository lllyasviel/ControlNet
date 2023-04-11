#!/bin/bash

def_cont_name=controlnet_cont

helpFunction()
{
   echo ""
   echo "Usage: $0 -p port"
   echo -e "\t-p http_port"
   exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
   case "$opt" in
      p ) port="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$port" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Check if the container is running
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    # Stop the container
    echo "Stopping docker container '$def_cont_name'"
    docker stop "$def_cont_name"
    echo "Stopped container '$def_cont_name'"
fi

# 7860 is gradio's default port
docker run \
      -ti --rm \
      -p "0.0.0.0:$port:7860" \
      -v "$PWD/models:/home/user1/ControlNet/models" \
      --gpus "device=0" \
      --name "$def_cont_name" \
      controlnet:latest
