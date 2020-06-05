#!/bin/bash
TARGET=/home/ubuntu/Emotional-Training/data/Saved_Files
PROCESSED=/home/ubuntu/Emotional-Training/data/live_images
inotifywait -m -e create -e moved_to --format "%f" $TARGET \
        | while read FILENAME
                do
                        echo Detected $FILENAME
                        python myfile.py
                done