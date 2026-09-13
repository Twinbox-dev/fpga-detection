import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def json_to_xml(json_path, xml_path):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建XML根元素
    annotation = ET.Element("annotation")
    
    # 添加文件夹和文件名信息
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = data.get("imagePath", "")
    
    # 添加图片尺寸信息
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(data.get("imageWidth", 0))
    height = ET.SubElement(size, "height")
    height.text = str(data.get("imageHeight", 0))
    depth = ET.SubElement(size, "depth")
    depth.text = "1"  # 默认为灰度图片
    
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    # 处理每个标注对象
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue  # 只处理矩形标注
        
        points = shape.get("points", [])
        if len(points) != 2:
            continue
        
        # 提取坐标
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # 确保xmin<xmax, ymin<ymax
        xmin_val = min(x1, x2)
        xmax_val = max(x1, x2)
        ymin_val = min(y1, y2)
        ymax_val = max(y1, y2)
        
        # 创建object元素
        obj = ET.SubElement(annotation, "object")
        
        name = ET.SubElement(obj, "name")
        name.text = shape.get("label", "")
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        # 创建边界框
        bndbox = ET.SubElement(obj, "bndbox")
        
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(round(xmin_val))
        
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(round(ymin_val))
        
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(round(xmax_val))
        
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(round(ymax_val))
    
    # 格式化并保存XML
    tree = ET.ElementTree(annotation)
    
    # 手动创建格式化字符串
    rough_string = ET.tostring(annotation, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="\t")  # 使用制表符缩进
    
    # 去掉XML声明行
    lines = pretty_xml.split('\n')
    if lines and lines[0].startswith('<?xml'):
    	lines = lines[1:]  # 去掉第一行
    
    # 重新组合
    xml_str = '\n'.join(lines)
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"转换完成：{xml_path}")

# 批量转换函数
def batch_convert(json_dir, xml_dir):
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)
    
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            xml_filename = filename.replace('.json', '.xml')
            xml_path = os.path.join(xml_dir, xml_filename)
            
            json_to_xml(json_path, xml_path)

# 使用示例
if __name__ == "__main__":
    # 单个文件转换
    # json_to_xml("ca_shang44.json", "ca_shang44.xml")
    
    # 批量转换
    # batch_convert("json_annotations/", "xml_annotations/")
	
    batch_convert("./","./xml")