#!/bin/bash
source ~/.poetry/env
poetry install
poetry run python api.py
