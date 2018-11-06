In LoadCOCO, I extract data from COCO data in order

**** STEP 1: From the annotation, generate new_annotation that only include the objects you want
             (python3 generate_new_annotation.py name_original_file.json object1 object2 ...)
    
**** STEP 2: From the new json you have, generate new_data (maybe extracted from Dataset) 
             (python3 generate_new_data.py)

**** STEP 3: If you have had the data you want, generate trainvalno5k.txt(fixed path to train images) and 5k.txt(fixed path to val images)
             python3 generate_new_5k_and_traivalno5k.py

**** STEP 4: Finally, you have to generate your labels folder
             If path Dataset/COCO/train2017/XYZ.jpg exists, labels/train2017/XYZ.txt have exist also
             XYZ.txt will generate normalize data from annotation bounding box and category_id
            
             E.g: Image :123456.jpg has dog and cat
             ===> 123456.txt has:
                <dog_id-1> <dog_bbox_width> <dog_bbox_height> <dog_x_center> <dog_y_center>
                <cat_id-1> <cat_bbox_width> <cat_bbox_height> <cat_x_center> <cat_y_center>
