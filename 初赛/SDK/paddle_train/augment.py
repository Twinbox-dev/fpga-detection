import os
import cv2
import numpy as np
from lxml import etree
import random

def parse_xml(xml_path):
    """解析XML，获取图片尺寸和标注框"""
    # 处理中文路径
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    root = etree.fromstring(xml_content.encode('utf-8'))
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])
    
    return root, w, h, bboxes

def save_xml(tree, save_path, new_bboxes, img_filename, augmentation_type=None):
    """保存新的XML标注，可选添加增强类型注释"""
    # 创建新的XML结构
    annotation = etree.Element("annotation")
    
    # 添加filename
    filename_elem = etree.SubElement(annotation, "filename")
    filename_elem.text = img_filename
    
    # 添加可选的增强类型注释
    if augmentation_type:
        augmentation_elem = etree.SubElement(annotation, "augmentation")
        augmentation_elem.text = augmentation_type
    
    # 添加size
    size_elem = etree.SubElement(annotation, "size")
    width_elem = etree.SubElement(size_elem, "width")
    height_elem = etree.SubElement(size_elem, "height")
    depth_elem = etree.SubElement(size_elem, "depth")
    
    # 这里需要从原tree获取size信息
    width_elem.text = tree.find("size/width").text
    height_elem.text = tree.find("size/height").text
    depth_elem.text = tree.find("size/depth").text if tree.find("size/depth") is not None else "1"
    
    # 添加标注框
    for idx, bbox in enumerate(new_bboxes):
        obj_elem = etree.SubElement(annotation, "object")
        
        name_elem = etree.SubElement(obj_elem, "name")
        name_elem.text = tree.findall("object")[idx].find("name").text
        
        bndbox_elem = etree.SubElement(obj_elem, "bndbox")
        xmin_elem = etree.SubElement(bndbox_elem, "xmin")
        ymin_elem = etree.SubElement(bndbox_elem, "ymin")
        xmax_elem = etree.SubElement(bndbox_elem, "xmax")
        ymax_elem = etree.SubElement(bndbox_elem, "ymax")
        
        xmin, ymin, xmax, ymax = bbox
        xmin_elem.text = str(int(xmin))
        ymin_elem.text = str(int(ymin))
        xmax_elem.text = str(int(xmax))
        ymax_elem.text = str(int(ymax))
    
    # 保存XML，处理中文路径
    tree = etree.ElementTree(annotation)
    tree.write(save_path, encoding='utf-8', pretty_print=True, xml_declaration=True)

# 基础增强函数 - 每个函数都返回操作名称
def horizontal_flip(img, bboxes, w, h):
    """水平翻转"""
    img = cv2.flip(img, 1)
    new_bboxes = [[w - xmax, ymin, w - xmin, ymax] for xmin, ymin, xmax, ymax in bboxes]
    return img, new_bboxes, "horizontal_flip"

def vertical_flip(img, bboxes, w, h):
    """垂直翻转"""
    img = cv2.flip(img, 0)
    new_bboxes = [[xmin, h - ymax, xmax, h - ymin] for xmin, ymin, xmax, ymax in bboxes]
    return img, new_bboxes, "vertical_flip"

def rotate_10(img, bboxes, w, h):
    """旋转10度"""
    angle = 10
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        transformed_points = np.dot(M, np.vstack([points.T, np.ones(4)])).T
        new_xmin, new_xmax = min(transformed_points[:, 0]), max(transformed_points[:, 0])
        new_ymin, new_ymax = min(transformed_points[:, 1]), max(transformed_points[:, 1])
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img, new_bboxes, "rotate_10"

def rotate_minus_10(img, bboxes, w, h):
    """旋转-10度"""
    angle = -10
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        transformed_points = np.dot(M, np.vstack([points.T, np.ones(4)])).T
        new_xmin, new_xmax = min(transformed_points[:, 0]), max(transformed_points[:, 0])
        new_ymin, new_ymax = min(transformed_points[:, 1]), max(transformed_points[:, 1])
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img, new_bboxes, "rotate_minus_10"

def sharpen(img, bboxes, w, h):
    """锐化增强"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img, bboxes, "sharpen"

def random_brightness(img, bboxes, w, h):
    """随机亮度调整"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + random.randint(-30, 30), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    return img, bboxes, "random_brightness"

def random_contrast(img, bboxes, w, h):
    """随机对比度调整"""
    img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=0)
    return img, bboxes, "random_contrast"

def random_blur(img, bboxes, w, h):
    """随机模糊处理"""
    img = cv2.GaussianBlur(img, (random.choice([3, 5, 7]), random.choice([3, 5, 7])), 0)
    return img, bboxes, "random_blur"

# 操作名称到中文描述的映射
AUG_NAME_MAP = {
    "horizontal_flip": "水平翻转",
    "vertical_flip": "垂直翻转", 
    "rotate_10": "旋转10度",
    "rotate_minus_10": "旋转-10度",
    "sharpen": "锐化处理",
    "random_brightness": "亮度调整",
    "random_contrast": "对比度调整",
    "random_blur": "模糊处理"
}

def augment_single(img_path, xml_path, output_dir, num_aug=5, multi_aug=False):
    """单张图片增强"""
    # 读取图片 - 处理中文路径
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"读取失败: {img_path}")
        return
    
    tree, w, h, bboxes = parse_xml(xml_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 创建输出目录 - 处理中文路径
    img_dir = os.path.join(output_dir, "images")
    xml_dir = os.path.join(output_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    
    # 增强操作列表
    aug_funcs = [
        horizontal_flip,
        vertical_flip,
        rotate_10,
        rotate_minus_10,
        sharpen,
        random_brightness,
        random_contrast,
        random_blur
    ]
    
    for i in range(num_aug):
        aug_img = img.copy()
        aug_bboxes = bboxes.copy()
        applied_ops = []  # 存储应用的操作名称
        applied_op_names = []  # 存储中文操作名称
        
        if multi_aug:
            # 多效果叠加：随机选择1-3个操作
            selected_funcs = random.sample(aug_funcs, random.randint(1, 3))
            for func in selected_funcs:
                aug_img, aug_bboxes, op_name = func(aug_img, aug_bboxes, w, h)
                applied_ops.append(op_name)
                applied_op_names.append(AUG_NAME_MAP[op_name])
        else:
            # 单效果：只应用一个操作
            func = random.choice(aug_funcs)
            aug_img, aug_bboxes, op_name = func(aug_img, aug_bboxes, w, h)
            applied_ops.append(op_name)
            applied_op_names.append(AUG_NAME_MAP[op_name])
        
        # 保存结果 - 处理中文路径
        aug_img_name = f"{base_name}_aug{i+1}.jpg"
        aug_xml_name = f"{base_name}_aug{i+1}.xml"
        
        # 保存图片
        img_save_path = os.path.join(img_dir, aug_img_name)
        cv2.imencode('.jpg', aug_img)[1].tofile(img_save_path)
        
        # 准备XML中的增强类型描述
        if multi_aug:
            xml_aug_desc = f"multi_aug:{','.join(applied_ops)}"
            display_desc = f"叠加({','.join(applied_op_names)})"
        else:
            xml_aug_desc = applied_ops[0]
            display_desc = applied_op_names[0]  # 单效果显示具体操作名称
        
        # 保存XML，包含增强类型信息
        xml_save_path = os.path.join(xml_dir, aug_xml_name)
        save_xml(tree, xml_save_path, aug_bboxes, aug_img_name, xml_aug_desc)
        
        print(f"生成: {aug_img_name} ({display_desc})")

if __name__ == "__main__":
    # 输入设置
    image_folder = "./images" # input("图片文件夹路径: ").strip()
    xml_folder = "./annotations" # input("XML标注文件夹路径: ").strip()
    output_dir = "./"
    num_aug = 9
    
    # 多效果叠加选项
    use_multi_aug = input("启用多效果叠加? (y/n, 默认n): ").strip().lower()
    multi_aug = use_multi_aug == 'y'
    
    print(f"开始批量增强...")
    print(f"图片文件夹: {image_folder}")
    print(f"多效果叠加: {'是' if multi_aug else '否'}")
    
    # 遍历文件夹
    processed = 0
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
            
        img_path = os.path.join(image_folder, filename)
        xml_path = os.path.join(xml_folder, os.path.splitext(filename)[0] + ".xml")
        
        if not os.path.exists(xml_path):
            print(f"跳过: {filename} (无对应标注)")
            continue
            
        augment_single(img_path, xml_path, output_dir, num_aug, multi_aug)
        processed += 1
    
    print(f"完成! 已处理 {processed} 张图片")