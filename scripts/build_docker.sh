#!/bin/bash

docker build -t controlnet:latest --build-arg UID=$(id -u) .
