#!/usr/bin/python3
import cv2
import numpy as np
import json
import sys

# cat_names la 1 mang chua tat ca cac categories ma chung ta muon luu tru lai
# xoa cac category ta khong can va tra ve cac index cua cac category
class OWN_COCO:
    def __init__(self, annotationFile = None):
        self.dataset = dict()
        self.newown = dict()
        if not annotationFile == None:
            print ("Loading dataset")
            self.dataset = json.load(open(annotationFile, 'r'))
            print ("Done!")
            for key in self.dataset:
                if key not in ['categories', 'images', 'annotations']:
                    self.newown[key] = self.dataset[key]
                else:
                    self.newown[key] = list()

    def chooseCategory(self, cat_names):
        cat_indices= []
        assert ('categories' in self.dataset), "You should have categories field"
        for cat in self.dataset['categories']:
            if cat['name'] in cat_names:
                self.newown['categories'].append(cat) 
                cat_indices.append(cat['id'])
        if len(cat_indices) == 0:
            print ("You must choose the name of category properly")
        return cat_indices
    
    # ta luu giu lai annotation chi cua nhung category ta can
    # tra ve cac image co chua cat_indices
    def chooseAnnotation(self, cat_indices):
        image_indices = []
        assert ('annotations' in self.dataset), "You should have annotations field"
        for ann in self.dataset['annotations']:
            if ann['category_id'] in cat_indices:
                self.newown['annotations'].append(ann)
                image_indices.append(ann['image_id'])
        # mac du dieu nay co the khong baoh xay ra
        if len(image_indices) == 0:
            print ("No image has categories you need")
        return image_indices
    
    # ta chi luu giu lai images voi category ta can:
    def chooseImages(self, image_indices):
        assert ('categories' in self.dataset), "You should have images field"
        for img in self.dataset['images']:
            if img['id'] in image_indices:
                self.newown['images'].append(img)
    
    # optional: Neu ta chi chon 1 loai, thi ta se phai select random trong cho anh con lai va gan label cho no la other
    #def otherLabel(self, cat_name):
    #    for ann in self.dataset["annotation"]:
    #        ann['category_id'] = 
            
    # ta con phai danh so lai category cua chung ta, giu nguyen name, super category, chi thay doi id
    # chung ta nen dung 1 dictionary de save lai "name": "id"
    def create_new_annotation_file_based_on_cat_we_choose(self, cat_names, output_name):
        cat_indices = self.chooseCategory(cat_names)
        image_indices = self.chooseAnnotation(cat_indices)
        self.chooseImages(image_indices)
        with open('./annotations/' + output_name, 'w') as outfile:
            json.dump(self.newown, outfile, indent=4, sort_keys=True)

def main(argv):
    assert (len(argv) > 2), "You should parse json file and categories you want to save"
    originalDataDir = "../GlobalData/COCO-Data/annotations/"
    annFile = originalDataDir + argv[1]
    cat_names = argv[2:]
    output_name= argv[1].split('.')[0]+'.json'
    print (annFile)
    print (cat_names)
    print (output_name)
    coco = OWN_COCO(annFile)
    coco.create_new_annotation_file_based_on_cat_we_choose(cat_names, output_name)

if __name__ == "__main__":
    main(sys.argv) 
