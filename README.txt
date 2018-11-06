In Dataset:
    *** You can use commandLine to download image data. For example, in order to download COCO2014:
            wget -c https://pjreddie.com/media/files/train2014.zip
            wget -c https://pjreddie.com/media/files/val2014.zip
            unzip -q train2014.zip
            unzip -q val2014.zip
    *** But the sizes of these data are really big, I recommend you choose an download manager, and then unzip it

In Dev:
    We will extract data from Dataset. 
    In this project, we extract from COCO data. You can also do it in VOC data by yourself

In Network_model:
    We have to reconfig Network if you want to train your own dataset
    In this project, I add my-person-yolo.cfg (modified from yolo.cfg)
                     I add my-person.data (modified from coco.data)
                     I add my-person.names (modified from coco.names)
