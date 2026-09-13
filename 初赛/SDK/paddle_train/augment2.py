import os
import cv2
import numpy as np
from lxml import etree
import random
import argparse
import shutil

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
    depth_elem.text = tree.find("size/depth").text if tree.find("size/depth") is not None else "3"
    
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
        xmin_elem.text = str(int(max(0, xmin)))
        ymin_elem.text = str(int(max(0, ymin)))
        xmax_elem.text = str(int(min(int(width_elem.text), xmax)))
        ymax_elem.text = str(int(min(int(height_elem.text), ymax)))
    
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

def rotate_30(img, bboxes, w, h):
    """旋转30度"""
    angle = 30
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        transformed_points = np.dot(M, np.vstack([points.T, np.ones(4)])).T
        new_xmin, new_xmax = min(transformed_points[:, 0]), max(transformed_points[:, 0])
        new_ymin, new_ymax = min(transformed_points[:, 1]), max(transformed_points[:, 1])
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img, new_bboxes, "rotate_30"

def rotate_minus_30(img, bboxes, w, h):
    """旋转-30度"""
    angle = -30
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        transformed_points = np.dot(M, np.vstack([points.T, np.ones(4)])).T
        new_xmin, new_xmax = min(transformed_points[:, 0]), max(transformed_points[:, 0])
        new_ymin, new_ymax = min(transformed_points[:, 1]), max(transformed_points[:, 1])
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img, new_bboxes, "rotate_minus_30"

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

def random_noise(img, bboxes, w, h):
    """添加随机噪声"""
    noise = np.random.normal(0, random.uniform(0, 25), img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return img, bboxes, "random_noise"

def random_saturation(img, bboxes, w, h):
    """随机饱和度调整"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.int16) + random.randint(-20, 20), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    return img, bboxes, "random_saturation"

def random_hue(img, bboxes, w, h):
    """随机色相调整"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = (h.astype(np.int16) + random.randint(-5, 5)) % 180
    h = h.astype(np.uint8)
    img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    return img, bboxes, "random_hue"

def crop_and_zoom(img, bboxes, w, h):
    """随机裁剪并缩放"""
    # 随机裁剪尺寸
    crop_w = int(w * random.uniform(0.8, 0.95))
    crop_h = int(h * random.uniform(0.8, 0.95))
    
    # 随机裁剪位置
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    
    # 裁剪图片
    img_cropped = img[y:y+crop_h, x:x+crop_w]
    
    # 缩放回原尺寸
    img_resized = cv2.resize(img_cropped, (w, h))
    
    # 调整标注框
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        # 将坐标映射到裁剪区域
        new_xmin = (xmin - x) * w / crop_w
        new_ymin = (ymin - y) * h / crop_h
        new_xmax = (xmax - x) * w / crop_w
        new_ymax = (ymax - y) * h / crop_h
        
        # 确保坐标在图片范围内
        new_xmin = max(0, new_xmin)
        new_ymin = max(0, new_ymin)
        new_xmax = min(w, new_xmax)
        new_ymax = min(h, new_ymax)
        
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img_resized, new_bboxes, "crop_and_zoom"

def perspective_transform(img, bboxes, w, h):
    """透视变换"""
    # 随机生成四个角点
    margin = int(min(w, h) * 0.1)
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    dx1 = random.randint(-margin, margin)
    dy1 = random.randint(-margin, margin)
    dx2 = random.randint(-margin, margin)
    dy2 = random.randint(-margin, margin)
    dx3 = random.randint(-margin, margin)
    dy3 = random.randint(-margin, margin)
    dx4 = random.randint(-margin, margin)
    dy4 = random.randint(-margin, margin)
    
    dst_points = np.float32([
        [dx1, dy1],
        [w + dx2, dy2],
        [w + dx3, h + dy3],
        [dx4, h + dy4]
    ])
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换
    img = cv2.warpPerspective(img, M, (w, h))
    
    # 调整标注框
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        # 变换四个角点
        points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, M)
        transformed_points = transformed_points.reshape(-1, 2)
        
        new_xmin = min(transformed_points[:, 0])
        new_ymin = min(transformed_points[:, 1])
        new_xmax = max(transformed_points[:, 0])
        new_ymax = max(transformed_points[:, 1])
        
        # 确保坐标在图片范围内
        new_xmin = max(0, new_xmin)
        new_ymin = max(0, new_ymin)
        new_xmax = min(w, new_xmax)
        new_ymax = min(h, new_ymax)
        
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img, new_bboxes, "perspective_transform"

def random_scale(img, bboxes, w, h):
    """随机缩放"""
    scale = random.uniform(0.7, 1.3)  # 可调范围扩大一些
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图片
    img_scaled = cv2.resize(img, (new_w, new_h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        # 将坐标映射到缩放后的尺寸
        scaled_xmin = xmin * new_w / w
        scaled_ymin = ymin * new_h / h
        scaled_xmax = xmax * new_w / w
        scaled_ymax = ymax * new_h / h
        
        if scale > 1:
            # 缩小：从放大图中裁剪中心部分
            x_offset = (new_w - w) // 2
            y_offset = (new_h - h) // 2
            
            # 检查边界框是否在裁剪区域内
            if (scaled_xmax > x_offset and scaled_xmin < x_offset + w and
                scaled_ymax > y_offset and scaled_ymin < y_offset + h):
                
                # 计算裁剪后的坐标
                clipped_xmin = max(0, scaled_xmin - x_offset)
                clipped_xmax = min(w, scaled_xmax - x_offset)
                clipped_ymin = max(0, scaled_ymin - y_offset)
                clipped_ymax = min(h, scaled_ymax - y_offset)
                
                # 确保边界框有效
                if clipped_xmin < clipped_xmax and clipped_ymin < clipped_ymax:
                    new_bboxes.append([clipped_xmin, clipped_ymin, 
                                       clipped_xmax, clipped_ymax])
        else:
            # 放大：填充黑色边框
            x_offset = (w - new_w) // 2
            y_offset = (h - new_h) // 2
            
            new_xmin = scaled_xmin + x_offset
            new_ymin = scaled_ymin + y_offset
            new_xmax = scaled_xmax + x_offset
            new_ymax = scaled_ymax + y_offset
            
            # 确保坐标在图片范围内
            new_xmin = max(0, new_xmin)
            new_ymin = max(0, new_ymin)
            new_xmax = min(w, new_xmax)
            new_ymax = min(h, new_ymax)
            
            if new_xmin < new_xmax and new_ymin < new_ymax:
                new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    # 调整图片到原尺寸
    if scale > 1:
        # 从放大图中裁剪中心部分
        x_start = (new_w - w) // 2
        y_start = (new_h - h) // 2
        img = img_scaled[y_start:y_start+h, x_start:x_start+w]
    else:
        # 填充黑色边框
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        new_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_scaled
        img = new_img
    
    return img, new_bboxes, "random_scale"

def grayscale(img, bboxes, w, h):
    """灰度化"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    return img, bboxes, "grayscale"

def add_snow_effect(img, bboxes, w, h):
    """添加雪花效果"""
    snow = np.random.randint(0, 255, (h, w, 1), dtype=np.uint8)
    snow_mask = (snow > 250).astype(np.uint8) * 255
    snow_mask = np.repeat(snow_mask, 3, axis=2)
    img = cv2.addWeighted(img, 0.9, snow_mask.astype(np.uint8), 0.3, 0)
    return img, bboxes, "snow_effect"

def add_rain_effect(img, bboxes, w, h):
    """添加雨滴效果"""
    rain = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(100):
        x = random.randint(0, w-1)
        length = random.randint(5, 20)
        width = random.randint(1, 2)
        cv2.line(rain, (x, 0), (x, length), (200, 200, 200), width)
    
    img = cv2.addWeighted(img, 0.9, rain, 0.2, 0)
    return img, bboxes, "rain_effect"

def motion_blur(img, bboxes, w, h):
    """运动模糊"""
    size = random.randint(5, 15)
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    img = cv2.filter2D(img, -1, kernel)
    return img, bboxes, "motion_blur"

def edge_enhance(img, bboxes, w, h):
    """边缘增强"""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return img, bboxes, "edge_enhance"

def horizontal_stretch(img, bboxes, w, h):
    """横向拉伸"""
    # 拉伸系数，可调范围
    stretch_factor = random.uniform(0.7, 1.3)
    new_w = int(w * stretch_factor)
    
    # 拉伸图片
    img_stretched = cv2.resize(img, (new_w, h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        # 将坐标映射到新尺寸
        new_xmin = xmin * new_w / w
        new_ymin = ymin
        new_xmax = xmax * new_w / w
        new_ymax = ymax
        
        if stretch_factor > 1:
            # 拉伸后宽度更大，裁剪到原始尺寸
            x_offset = (new_w - w) // 2
            # 裁剪后，检查边界框是否在可见区域内
            if new_xmax > x_offset and new_xmin < x_offset + w:
                clipped_xmin = max(0, new_xmin - x_offset)
                clipped_xmax = min(w, new_xmax - x_offset)
                new_bboxes.append([clipped_xmin, new_ymin, clipped_xmax, new_ymax])
        else:
            # 拉伸后宽度更小，填充黑色边框
            x_offset = (w - new_w) // 2
            new_bboxes.append([new_xmin + x_offset, new_ymin, 
                             new_xmax + x_offset, new_ymax])
    
    # 调整图片到原尺寸
    if stretch_factor > 1:
        # 裁剪中心部分
        x_start = (new_w - w) // 2
        img = img_stretched[:, x_start:x_start+w]
    else:
        # 填充黑色边框
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        x_offset = (w - new_w) // 2
        new_img[:, x_offset:x_offset+new_w] = img_stretched
        img = new_img
    
    return img, new_bboxes, "horizontal_stretch"


def vertical_stretch(img, bboxes, w, h):
    """竖向拉伸"""
    # 拉伸系数，可调范围
    stretch_factor = random.uniform(0.7, 1.3)
    new_h = int(h * stretch_factor)
    
    # 拉伸图片
    img_stretched = cv2.resize(img, (w, new_h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        # 将坐标映射到新尺寸
        new_xmin = xmin
        new_ymin = ymin * new_h / h
        new_xmax = xmax
        new_ymax = ymax * new_h / h
        
        if stretch_factor > 1:
            # 拉伸后高度更大，裁剪到原始尺寸
            y_offset = (new_h - h) // 2
            # 裁剪后，检查边界框是否在可见区域内
            if new_ymax > y_offset and new_ymin < y_offset + h:
                clipped_ymin = max(0, new_ymin - y_offset)
                clipped_ymax = min(h, new_ymax - y_offset)
                new_bboxes.append([new_xmin, clipped_ymin, new_xmax, clipped_ymax])
        else:
            # 拉伸后高度更小，填充黑色边框
            y_offset = (h - new_h) // 2
            new_bboxes.append([new_xmin, new_ymin + y_offset,
                             new_xmax, new_ymax + y_offset])
    
    # 调整图片到原尺寸
    if stretch_factor > 1:
        # 裁剪中心部分
        y_start = (new_h - h) // 2
        img = img_stretched[y_start:y_start+h, :]
    else:
        # 填充黑色边框
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        y_offset = (h - new_h) // 2
        new_img[y_offset:y_offset+new_h, :] = img_stretched
        img = new_img
    
    return img, new_bboxes, "vertical_stretch"


# 操作名称到中文描述的映射
AUG_NAME_MAP = {
    "horizontal_flip": "水平翻转",
    "vertical_flip": "垂直翻转", 
    "rotate_10": "旋转10度",
    "rotate_minus_10": "旋转-10度",
    "rotate_30": "旋转30度",
    "rotate_minus_30": "旋转-30度",
    "sharpen": "锐化处理",
    "random_brightness": "亮度调整",
    "random_contrast": "对比度调整",
    "random_blur": "模糊处理",
    "random_noise": "随机噪声",
    "random_saturation": "饱和度调整",
    "random_hue": "色相调整",
    "crop_and_zoom": "裁剪缩放",
    "perspective_transform": "透视变换",
    "random_scale": "随机缩放",
    "grayscale": "灰度化",
    "snow_effect": "雪花效果",
    "rain_effect": "雨滴效果",
    "motion_blur": "运动模糊",
    "edge_enhance": "边缘增强",
    "horizontal_stretch": "横向拉伸",
    "vertical_stretch": "竖向拉伸"
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
        rotate_30,
        rotate_minus_30,
        sharpen,
        random_brightness,
        random_contrast,
        random_blur,
        random_noise,
        random_saturation,
        random_hue,
        crop_and_zoom,
        perspective_transform,
        random_scale,
        grayscale,
        add_snow_effect,
        add_rain_effect,
        motion_blur,
        edge_enhance,
        horizontal_stretch,
        vertical_stretch
    ]
    
    for i in range(num_aug):
        aug_img = img.copy()
        aug_bboxes = bboxes.copy()
        applied_ops = []  # 存储应用的操作名称
        applied_op_names = []  # 存储中文操作名称
        
        if multi_aug:
            # 多效果叠加：随机选择1-4个操作
            selected_funcs = random.sample(aug_funcs, random.randint(1, 4))
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

def clear_augmented_data(output_dir):
    """清除增强数据"""
    img_dir = os.path.join(output_dir, "images")
    xml_dir = os.path.join(output_dir, "annotations")
    
    files_removed = 0
    
    # 删除images文件夹中的增强图片
    if os.path.exists(img_dir):
        for filename in os.listdir(img_dir):
            if filename.endswith('_aug') or '_aug' in filename:
                file_path = os.path.join(img_dir, filename)
                os.remove(file_path)
                print(f"删除: {file_path}")
                files_removed += 1
    
    # 删除annotations文件夹中的增强XML
    if os.path.exists(xml_dir):
        for filename in os.listdir(xml_dir):
            if filename.endswith('_aug') or '_aug' in filename:
                file_path = os.path.join(xml_dir, filename)
                os.remove(file_path)
                print(f"删除: {file_path}")
                files_removed += 1
    
    print(f"已删除 {files_removed} 个增强文件")

def main():
    parser = argparse.ArgumentParser(
        description="图片数据增强工具 - 支持22种增强方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python data_augmentation.py                          # 交互式运行
  python data_augmentation.py -m                       # 启用多效果叠加
  python data_augmentation.py -i ./images -x ./annotations -n 5
  python data_augmentation.py -C                       # 清除增强数据
  python data_augmentation.py --img image.jpg --xml annotation.xml  # 单张图片处理
  
支持23种增强方法:
  1. 水平翻转    2. 垂直翻转    3. 旋转10度    4. 旋转-10度
  5. 旋转30度     6. 旋转-30度    7. 锐化处理    8. 亮度调整
  9. 对比度调整 10. 模糊处理    11. 随机噪声   12. 饱和度调整
  13. 色相调整  14. 裁剪缩放    15. 透视变换   16. 随机缩放
  17. 灰度化    18. 雪花效果    19. 雨滴效果   20. 运动模糊
  21. 边缘增强  22. 横向拉伸    23. 竖向拉伸
"""
    )
    
    parser.add_argument("--image_folder", "-i", default="./images", help="图片文件夹路径，默认: ./images")
    parser.add_argument("--xml_folder", "-x", default="./annotations", help="XML标注文件夹路径，默认: ./annotations")
    parser.add_argument("--output_dir", "-o", default="./", help="输出目录，默认: 当前目录")
    parser.add_argument("--num_aug", "-n", type=int, default=9, help="每张图片生成增强图片数量，默认: 9")
    parser.add_argument("--multi_aug", "-m", action="store_true", help="启用多效果叠加")
    parser.add_argument("--clear", "-C", action="store_true", help="清除增强数据而不生成新数据")
    
    # 新增 --img 参数用于单张图片处理
    parser.add_argument("--img", help="单张图片路径，与--xml参数配合使用")
    parser.add_argument("--xml", help="单张图片对应的XML标注路径，与--img参数配合使用")
    
    args = parser.parse_args()
    
    if args.clear:
        print(f"正在清除增强数据...")
        clear_augmented_data(args.output_dir)
        return
    
    # 检查是单张图片处理还是批量处理
    if args.img is not None or args.xml is not None:
        # 单张图片处理模式
        if args.img is None or args.xml is None:
            print("错误: 单张图片处理需要同时指定 --img 和 --xml 参数")
            return
        
        if not os.path.exists(args.img):
            print(f"错误: 图片文件不存在: {args.img}")
            return
        
        if not os.path.exists(args.xml):
            print(f"错误: XML文件不存在: {args.xml}")
            return
        
        print(f"开始单张图片增强...")
        print(f"图片: {args.img}")
        print(f"标注: {args.xml}")
        print(f"输出目录: {args.output_dir}")
        print(f"增强数量: {args.num_aug}")
        print(f"多效果叠加: {'是' if args.multi_aug else '否'}")
        
        augment_single(args.img, args.xml, args.output_dir, args.num_aug, args.multi_aug)
        print(f"完成! 已处理 1 张图片")
    
    else:
        # 批量处理模式（原有逻辑）
        print(f"开始批量增强...")
        print(f"图片文件夹: {args.image_folder}")
        print(f"XML文件夹: {args.xml_folder}")
        print(f"输出目录: {args.output_dir}")
        print(f"每张图片增强数量: {args.num_aug}")
        print(f"多效果叠加: {'是' if args.multi_aug else '否'}")
        print(f"支持增强方法数量: 22种")
        
        # 遍历文件夹
        processed = 0
        for filename in os.listdir(args.image_folder):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            img_path = os.path.join(args.image_folder, filename)
            xml_path = os.path.join(args.xml_folder, os.path.splitext(filename)[0] + ".xml")
            
            if not os.path.exists(xml_path):
                print(f"跳过: {filename} (无对应标注)")
                continue
                
            augment_single(img_path, xml_path, args.output_dir, args.num_aug, args.multi_aug)
            processed += 1
        
        print(f"完成! 已处理 {processed} 张图片 -> {processed*args.num_aug}张增强图片")

if __name__ == "__main__":
    main()