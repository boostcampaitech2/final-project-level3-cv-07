#!/bin/bash
source ~/.poetry/env
poetry install
poetry run poe force-cuda11