#!/bin/bash

cd darknet
cd data

# Clone COCO API
#git clone https://github.com/pdollar/coco
cd coco
cp ../../../../Dev/LoadCOCO/*.py .

# Generate Annotation
./generate_new_json.py instances_train2017.json person
./generate_new_json.py instances_val2017.json person
echo "****************GENERATE JSON DONE****************"

# Get Image    
mkdir images
cd images

mv ../generate_new_data.py .
./generate_new_data.py ../annotations/instances_train2017.json
./generate_new_data.py ../annotations/instances_val2017.json


echo "****************GENERATE DATA DONE****************"
mv generate_new_data.py ..

cd ..
# Get Your Own Metadata
./generate_new_labels.py instances_train2017.json
./generate_new_labels.py instances_val2017.json

echo "****************GENERATE LABELS  DONE****************"
./generate_new_trainvalno5k.py 

echo "****************GENERATE TRAINVALNO5K DONE*******************"
# Remove all script 
rm generate_new_*
