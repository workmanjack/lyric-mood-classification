#!/bin/bash

if [[ $1 == remote ]]; then
    jupyter-notebook --ip=0.0.0.0 --no-browser --port=5000
else
    jupyter-notebook
fi
