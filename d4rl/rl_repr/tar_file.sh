#!/bin/bash

find ./new_datas/ \( -name "*.csv" \) -print > files.txt
tar czvf data.tar.gz --files-from files.txt