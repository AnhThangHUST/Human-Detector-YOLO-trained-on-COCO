#!/bin/bash

cd darknet

# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

cp ./../Dev/LoadCOCO/*.py .

# Generate Annotation
./generat_new_json.py

# Get Image    
mkdir images
cd images

./generate_new_data.py

cd ..

# Get Your Own Metadata

./generate_new_trainvalno5k.py

./generate_new_labels.py

# Remove all script 
rm generat_new_*
