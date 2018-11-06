Before you use the darknet, you should config:
    darknet/cfg/yours.data          - E.g: darknet/cfg/my-person.data
    darknet/cfg/yours-network.cfg   - E.g: darknet/cfg/my-person-yolo.cfg
    darknet/data/yours.name         - E.g: darknet/data/my-person.names

Don't forget "make" inside the darknet

Instead of running darknet/scripts/get_coco_dataset.sh you should write your own script
E.g: you can test my_darknet_person_script.sh - using command: "bash my_darknet_person_script.sh"

Finally, you can use the following to train:
    ./darknet detector train cfg/my-person.data cfg/my-person-yolo.cfg darknet53.conv.74

