#!/usr/bin/python3
import json 
import os
import sys

originalDir = './annotations/'
destDir = './labels/'

# BBox la 1 list cua cac toa do chua dc chuan hoa, image_size la 1 list chua width va height
def NormalizedBBox(BBox, image_size):
    width, height = image_size
    BBox[0] = str(round(BBox[0]/width, 6))
    BBox[2] = str(round(BBox[2]/width, 6))
    BBox[1] = str(round(BBox[1]/height,6))
    BBox[3] = str(round(BBox[3]/height,6))
    return BBox

def process(annFile):
    print ("Start Generate My Labels from " + annFile)
    typeOfFile = ""
    if annFile.find('train') != -1:
        typeOfFile = annFile[annFile.find('train') : annFile.find('train') + 9]
    elif annFile.find('val') != -1:
        typeOfFile = annFile[annFile.find('val') : annFile.find('val') + 7]
    else:
        print ("You should configure your file is train or val?")
        return
    dataset = json.load(open(annFile,"r"))
    images = dict() # map image_id and_image name
    assert ('images' in dataset), "Your file must have images field"
    for image in dataset['images']:
        images[image['id']] = [image['file_name'], image['width'], image['height']]

    assert ("annotations" in dataset), "Your file must have annotations field"
    for ann in dataset['annotations']:
        nml_bbox = NormalizedBBox(ann['bbox'], images[ann['image_id']][1:3])
        cat = str(int(ann['category_id']) - 1)
        image_name = images[ann['image_id']][0]
        pathToFile = "./labels/" + typeOfFile + "/COCO_" + typeOfFile +"_" + image_name.split('.')[0] + ".txt"
        if not os.path.isdir("./labels"):
            os.mkdir("./labels")
        if not os.path.isdir("./labels/"+typeOfFile):
            os.mkdir("./labels/"+typeOfFile)
        if os.path.isfile(pathToFile):
            f = open(pathToFile, "a")
        else:
            f = open(pathToFile, "w")
        f.write(' '.join([cat] + nml_bbox)+"\n")

# xac dinh ro ouput la 1 file chua: catgories id
def main(argv):
    assert (len(argv)==2), "You have to add your annFile"
    annFile = argv[1]
    process(originalDir+annFile)

if __name__=='__main__':
    main(sys.argv)

