import xml.etree.ElementTree as ET
import json
import os
import base64
from PIL import Image
import cv2

def xml_to_json(xml_path, json_path, image_path=None):
    """
    将XML标注文件转换为Labelme格式的JSON文件
    
    参数:
    xml_path: XML文件路径
    json_path: 输出的JSON文件路径
    image_path: 对应的图片路径（可选，用于获取图片信息）
    """
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 创建JSON数据结构
    data = {
        "version": "5.1.1",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageData": None,
        "imageHeight": 0,
        "imageWidth": 0
    }
    
    # 提取基本信息
    for child in root:
        if child.tag == 'filename':
            data['imagePath'] = child.text
        elif child.tag == 'size':
            for size_child in child:
                if size_child.tag == 'width':
                    data['imageWidth'] = int(size_child.text)
                elif size_child.tag == 'height':
                    data['imageHeight'] = int(size_child.text)
    
    # 如果没有指定图片路径，尝试从XML中获取
    if image_path and os.path.exists(image_path):
        # 可以读取图片并将图片数据编码为base64
        with open(image_path, "rb") as image_file:
            data['imageData'] = base64.b64encode(image_file.read()).decode('utf-8')
    elif data['imagePath']:
        # 尝试在当前目录查找图片
        possible_path = os.path.join(os.path.dirname(xml_path), data['imagePath'])
        if os.path.exists(possible_path):
            with open(possible_path, "rb") as image_file:
                data['imageData'] = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 提取所有object（标注对象）
    for obj in root.findall('object'):
        # 获取标注信息
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        if bndbox is not None:
            # 提取坐标
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # 转换为Labelme格式的points
            # 矩形标注需要两个点：左上角和右下角
            points = [
                [float(xmin), float(ymin)],  # 左上角
                [float(xmax), float(ymax)]   # 右下角
            ]
            
            # 创建shape对象
            shape = {
                "label": name,
                "points": points,
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            
            data['shapes'].append(shape)
    
    # 保存为JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成：{json_path}")
    return data

def batch_xml_to_json(xml_dir, json_dir, image_dir=None):
    """
    批量将XML文件转换为JSON格式
    
    参数:
    xml_dir: XML文件目录
    json_dir: 输出的JSON文件目录
    image_dir: 图片文件目录（可选）
    """
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        json_file = xml_file.replace('.xml', '.json')
        json_path = os.path.join(json_dir, json_file)
        
        # 尝试查找对应的图片文件
        image_path = None
        if image_dir:
            # 从XML中获取文件名
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image_filename = root.find('filename').text
            if image_filename:
                possible_image_path = os.path.join(image_dir, image_filename)
                if os.path.exists(possible_image_path):
                    image_path = possible_image_path
        
        xml_to_json(xml_path, json_path, image_path)

# 使用示例
if __name__ == "__main__":
    # 单个文件转换
    # xml_to_json("ca_shang11.xml", "ca_shang11.json")
    
    # 如果图片在同一目录，可以传入图片路径
    # xml_to_json("ca_shang11.xml", "ca_shang11.json", "ca_shang11.jpg")
    
    # 批量转换示例
    # batch_xml_to_json("xml_annotations/", "json_annotations/", "images/")

	batch_xml_to_json("./","json")