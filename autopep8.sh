#!/bin/bash

set -e
set -x

dirs=('src' 'test')

for dir in ${dirs[@]}; do
    python3 -m autopep8 --in-place --recursive $dir
done
