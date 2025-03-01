#!/bin/bash

# if necessary, re-generate the log file
# python analytics.py --file nginx.log generate

# if the file analytics.db exists, remove it
if [ -f analytics.db ]; then
    rm analytics.db
fi
python analytics.py --file nginx.log write
python analytics.py read
python analytics.py --duration=8 averages
python analytics.py --attribute_count=3 prediction
