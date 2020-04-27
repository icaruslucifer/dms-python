# _*_ coding=UTF-8 _*_


import os,glob


paths = glob.glob("./phone/left_phonecall/pos/*.jpg")

for path in paths:
    filepath,filename = os.path.split(path)
    new_path = os.path.join(filepath,"left_"+filename)
    os.rename(path,new_path)