#!/usr/bin/python3
import os
import sys
import json
import shutil

originalDir = "../../../../../Dataset/COCO/"
destDir = "./"
def copy_only_image_we_need(annFile):
    dir_copy_from = None
    if annFile.find('train') != -1:
        dir_copy_from = annFile[annFile.find('train') : annFile.find('train') + 9]
    elif annFile.find('val') != -1:
        dir_copy_from = annFile[annFile.find('val') : annFile.find('val') + 7]
    else:
        print ("You should configure your file is train or val?")
        return
    dataset = json.load(open(annFile, 'r'))
    num_imgs = len(dataset['images'])
    dst = destDir + dir_copy_from
    print("Start copy image from: COCO"+ dir_copy_from)
    if not os.path.isdir(dst):
        os.mkdir(dst)
    for i, img in enumerate(dataset['images']):
        if (i%10000 == 0 and i !=0) or i==num_imgs-1:
            print ("=============> Copying Completed "+ str(round(float(i)*100/num_imgs)) +"%")
        src = originalDir + dir_copy_from + "/" + img['file_name']
        shutil.copy(src, dst)

def main(argv):
    assert (len(argv)==2), "You have to add your annFile"
    annFile = argv[1]
    copy_only_image_we_need(annFile)

if __name__=='__main__':
    main(sys.argv)
