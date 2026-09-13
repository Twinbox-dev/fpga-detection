import os
import cv2
import numpy as np
from lxml import etree
import random

def parse_xml(xml_path):
    """解析XML，获取图片尺寸和标注框"""
    tree = etree.parse(xml_path)
    root = tree.getroot()
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
    return tree, w, h, bboxes

def save_xml(tree, save_path, new_bboxes, img_filename):
    """保存新的XML标注，修复filename指向问题"""
    root = tree.getroot()
    
    # 修复：更新filename为正确的增强图片名
    root.find("filename").text = img_filename
    
    # 更新标注框坐标
    objects = root.findall("object")
    for idx, obj in enumerate(objects):
        bndbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = new_bboxes[idx]
        bndbox.find("xmin").text = str(int(xmin))
        bndbox.find("ymin").text = str(int(ymin))
        bndbox.find("xmax").text = str(int(xmax))
        bndbox.find("ymax").text = str(int(ymax))
    
    tree.write(save_path, encoding="utf-8", pretty_print=True)

# 所有增强函数统一为接受4个参数：(img, bboxes, w, h)
def horizontal_flip(img, bboxes, w, h):
    """水平翻转"""
    img = cv2.flip(img, 1)
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        new_xmin = w - xmax
        new_xmax = w - xmin
        new_bboxes.append([new_xmin, ymin, new_xmax, ymax])
    return img, new_bboxes, "水平翻转"

def vertical_flip(img, bboxes, w, h):
    """垂直翻转"""
    img = cv2.flip(img, 0)
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        new_ymin = h - ymax
        new_ymax = h - ymin
        new_bboxes.append([xmin, new_ymin, xmax, new_ymax])
    return img, new_bboxes, "垂直翻转"

def rotate_10(img, bboxes, w, h):
    """旋转10度"""
    angle = 10
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = M.dot(points_ones.T).T
        
        new_xmin = min(transformed_points[:, 0])
        new_ymin = min(transformed_points[:, 1])
        new_xmax = max(transformed_points[:, 0])
        new_ymax = max(transformed_points[:, 1])
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img, new_bboxes, "旋转10度"

def rotate_minus_10(img, bboxes, w, h):
    """旋转-10度"""
    angle = -10
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    new_bboxes = []
    for xmin, ymin, xmax, ymax in bboxes:
        points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = M.dot(points_ones.T).T
        
        new_xmin = min(transformed_points[:, 0])
        new_ymin = min(transformed_points[:, 1])
        new_xmax = max(transformed_points[:, 0])
        new_ymax = max(transformed_points[:, 1])
        new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return img, new_bboxes, "旋转-10度"

def sharpen(img, bboxes, w, h):
    """锐化增强"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img, bboxes, "锐化"

def random_brightness(img, bboxes, w, h):
    """随机亮度"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    val = random.randint(-30, 30)
    v = np.clip(v.astype(np.int16) + val, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img, bboxes, "亮度调整"

def random_contrast(img, bboxes, w, h):
    """随机对比度"""
    alpha = random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return img, bboxes, "对比度调整"

def random_blur(img, bboxes, w, h):
    """随机模糊"""
    k = random.choice([3, 5, 7])
    img = cv2.GaussianBlur(img, (k, k), 0)
    return img, bboxes, "模糊处理"

def augment_single_image(img_path, xml_path, output_dir, num_aug=5):
    """单张图片增强"""
    # 读取图片（使用cv2.imdecode处理中文路径）
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"读取失败: {img_path}")
        return
    
    # 统一为3通道BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # 读取标注
    try:
        tree, w, h, bboxes = parse_xml(xml_path)
    except Exception as e:
        print(f"解析XML失败: {xml_path}, 错误: {e}")
        return
    
    # 获取基础文件名
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    xml_dir = os.path.join(output_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    
    # 增强操作列表（每次只应用一个）
    aug_functions = [
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
        # 随机选择一个增强操作
        aug_func = random.choice(aug_functions)
        
        # 应用增强（所有函数现在都接受4个参数）
        aug_img, aug_bboxes, op_name = aug_func(img.copy(), bboxes.copy(), w, h)
        
        # 保存增强结果
        aug_img_name = f"{base_name}_aug{i+1}.jpg"
        aug_xml_name = f"{base_name}_aug{i+1}.xml"
        
        # 保存图片
        img_save_path = os.path.join(img_dir, aug_img_name)
        cv2.imwrite(img_save_path, aug_img)
        
        # 保存XML（修复filename指向）
        xml_save_path = os.path.join(xml_dir, aug_xml_name)
        save_xml(tree, xml_save_path, aug_bboxes, aug_img_name)
        
        print(f"生成: {aug_img_name} ({op_name})")

if __name__ == "__main__":
    # 只需要两个参数
    img_path = input("输入图片路径: ").strip()
    xml_path = input("输入XML标注路径: ").strip()
    output_dir = "./augmented_data"
    num_aug = 8
    
    print(f"开始增强: {img_path}")
    augment_single_image(img_path, xml_path, output_dir, num_aug)
    print(f"增强完成！结果保存在: {output_dir}")