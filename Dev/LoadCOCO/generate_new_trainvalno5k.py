#!/usr/bin/python3
import os

# generate 5k.part - validation or trainvalno5k.part

def generate(path):
    print ("Start generate trainvalno5k and 5k")
    files = os.listdir(path)
    if 'val' in path:
        f = open("./my-person-5k.txt", "w")
    elif 'train' in path:
        f = open("./my-person-trainvalno5k.txt", "w")
    else:
        print("You have to create train and validation dataset")
    for name in sorted(files):
        full_link = path + "/" + name +"\n"
        f.write(full_link)
    f.close()

def main():
    string = os.getcwd()
    originalDir = 'images/'
    fixed_path = string+'/'+originalDir
    folders = os.listdir(fixed_path)
    for folder in folders:
        generate(fixed_path+folder)

if __name__=="__main__":
    main()
    
