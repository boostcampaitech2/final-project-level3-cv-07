#!/bin/bash
git pull origin main
docker build . -t ocr
docker stop ocr
docker rm ocr
docker run --name ocr -it --shm-size=8G -d -p 80:6006 --ip=host ocr:latest