import os
import random
import xml.etree.ElementTree as ET 

trainval_percent = 0.95  #训练集验证集总占比
train_percent = 0.9  #训练集在trainval_percent里的train占比
xmlfilepath = './annotations'
txtsavepath = './ImageSet'
total_xml = os.listdir(xmlfilepath)

for filename in total_xml:  # 遍历所有xml文件
    tree = ET.parse(xmlfilepath + os.sep + filename)  
    root = tree.getroot()
    
    # 查找并删除path和source节点
    for elem in root.iter():
        if elem.tag == "path" or elem.tag == "source":
            parent = root.find(f'.//{elem.tag}/..')
            if parent is not None:
                parent.remove(elem)
    
    # 写入修改后的XML文件
    tree.write(xmlfilepath + os.sep + filename, encoding='utf-8')

num=len(total_xml)
list111=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list111,tv)
train=random.sample(trainval,tr)

ftrainval = open('./ImageSet/trainval.txt', 'w')
ftest = open('./ImageSet/test.txt', 'w')
ftrain = open('./ImageSet/train.txt', 'w')
fval = open('./ImageSet/val.txt', 'w')

for i  in list111:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()