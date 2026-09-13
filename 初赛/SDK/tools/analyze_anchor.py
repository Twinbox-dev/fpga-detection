import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_xml_annotation(xml_path, model_input_size=300):
    """解析单个XML标注文件，考虑原始图片和模型输入尺寸的差异"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取原始图片尺寸
    size_elem = root.find('size')
    img_width = int(size_elem.find('width').text)
    img_height = int(size_elem.find('height').text)
    
    # 计算缩放比例
    width_scale = model_input_size / img_width
    height_scale = model_input_size / img_height
    
    # 解析所有目标
    objects = []
    for obj_elem in root.findall('object'):
        name = obj_elem.find('name').text
        bbox_elem = obj_elem.find('bndbox')
        
        xmin = int(bbox_elem.find('xmin').text)
        ymin = int(bbox_elem.find('ymin').text)
        xmax = int(bbox_elem.find('xmax').text)
        ymax = int(bbox_elem.find('ymax').text)
        
        # 将坐标缩放到模型输入尺寸
        xmin_scaled = xmin * width_scale
        ymin_scaled = ymin * height_scale
        xmax_scaled = xmax * width_scale
        ymax_scaled = ymax * height_scale
        
        # 计算缩放后的宽度、高度、面积
        width = xmax_scaled - xmin_scaled
        height = ymax_scaled - ymin_scaled
        area = width * height
        
        objects.append({
            'class': name,
            'original_width': xmax - xmin,  # 原始尺寸
            'original_height': ymax - ymin,  # 原始尺寸
            'width': width,  # 缩放后的尺寸
            'height': height,  # 缩放后的尺寸
            'area': area,
            'aspect_ratio': width / height if height > 0 else 0,
            'img_width': img_width,
            'img_height': img_height
        })
    
    return objects

def analyze_dataset_and_calculate_anchors(xml_dir, model_input_size=300):
    """分析数据集并计算anchor配置，考虑图片缩放"""
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f"找到 {len(xml_files)} 个标注文件")
    print(f"模型输入尺寸: {model_input_size}×{model_input_size}")
    
    # 按类别存储统计信息
    class_stats = defaultdict(list)
    all_areas = []
    image_sizes = []
    
    for xml_file in xml_files:
        try:
            objects = parse_xml_annotation(xml_file, model_input_size)
            
            for obj in objects:
                class_name = obj['class']
                class_stats[class_name].append(obj)
                all_areas.append(obj['area'])
                # 记录图片尺寸用于统计
                image_sizes.append((obj['img_width'], obj['img_height']))
        except Exception as e:
            print(f"解析文件 {os.path.basename(xml_file)} 时出错: {e}")
    
    if not all_areas:
        print("错误: 没有找到任何目标框!")
        return None
    
    # 分析图片尺寸分布
    if image_sizes:
        unique_sizes = set(image_sizes)
        print(f"\n图片尺寸统计:")
        print(f"  不同尺寸数量: {len(unique_sizes)}")
        for i, (w, h) in enumerate(list(unique_sizes)[:5]):  # 显示前5种尺寸
            print(f"  尺寸{i+1}: {w}×{h}")
        if len(unique_sizes) > 5:
            print(f"  ... 还有{len(unique_sizes)-5}种其他尺寸")
    
    # 1. 显示类别统计（缩放后的尺寸）
    print("\n" + "="*60)
    print("各类别尺寸统计（缩放后）")
    print("="*60)
    
    for class_name, objects in class_stats.items():
        widths = [obj['width'] for obj in objects]  # 缩放后的宽度
        heights = [obj['height'] for obj in objects]  # 缩放后的高度
        areas = [obj['area'] for obj in objects]
        ratios = [obj['aspect_ratio'] for obj in objects]
        
        # 原始尺寸统计
        orig_widths = [obj['original_width'] for obj in objects]
        orig_heights = [obj['original_height'] for obj in objects]
        
        print(f"\n{class_name}:")
        print(f"  数量: {len(objects)}")
        print(f"  原始尺寸: {np.mean(orig_widths):.1f}×{np.mean(orig_heights):.1f} 像素")
        print(f"  缩放后尺寸: {np.mean(widths):.1f}×{np.mean(heights):.1f} 像素")
        print(f"  面积: {np.mean(areas):.1f}±{np.std(areas):.1f} 像素²")
        print(f"  宽高比: {np.mean(ratios):.2f}±{np.std(ratios):.2f}")
    
    # 2. 计算整体统计
    all_areas = np.array(all_areas)
    all_sizes = np.sqrt(all_areas)  # 面积转边长
    all_ratios = (all_sizes / model_input_size) * 100  # 转百分比
    
    print(f"\n整体统计（缩放后）:")
    print(f"  总目标数: {len(all_areas)}")
    print(f"  面积中位数: {np.median(all_areas):.1f} 像素²")
    print(f"  25%-75%面积: [{np.percentile(all_areas, 25):.1f}, {np.percentile(all_areas, 75):.1f}]")
    print(f"  最小尺寸: {np.min(all_sizes):.1f} 像素")
    print(f"  最大尺寸: {np.max(all_sizes):.1f} 像素")
    
    # 3. 计算min_sizes和max_sizes
    print(f"\n" + "="*60)
    print("推荐anchor配置")
    print("="*60)
    
    # 基于分位数计算6个层的尺寸
    min_sizes = []
    max_sizes = []
    
    # 使用不同的分位数策略
    quantiles = [0, 0.05, 0.15, 0.35, 0.55, 0.75, 1.0]  # 7个分界点对应6个区间
    
    for i in range(6):  # 6个层
        q_low = quantiles[i]
        q_high = quantiles[i+1]
        
        # 获取该区间的尺寸
        low_val = np.percentile(all_ratios, q_low * 100)
        high_val = np.percentile(all_ratios, q_high * 100)
        
        layer_sizes = all_ratios[(all_ratios >= low_val) & (all_ratios <= high_val)]
        
        if len(layer_sizes) > 0:
            min_size = max(0.5, np.percentile(layer_sizes, 10))  # 取10%分位
            if i < 5:  # 前5层有max_size
                max_size = min(100, np.percentile(layer_sizes, 90) * 1.2)  # 取90%分位*1.2
            else:  # 最后一层
                max_size = None
        else:
            # 备用计算
            base_size = 3.0 + i * 3.5
            min_size = base_size
            max_size = base_size * 1.8 if i < 5 else None
        
        # 使用Python原生float类型，避免np.float64
        min_sizes.append(float(round(min_size, 1)))
        if max_size is not None:
            max_sizes.append(float(round(max_size, 1)))
        else:
            max_sizes.append([])
    
    # 4. 计算宽高比配置
    all_aspect_ratios = []
    for objects in class_stats.values():
        all_aspect_ratios.extend([obj['aspect_ratio'] for obj in objects])
    
    all_aspect_ratios = np.array(all_aspect_ratios)
    
    # 找出常见的宽高比
    common_ratios = []
    for ratio in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        # 统计在ratio附近的目标比例
        count = np.sum((all_aspect_ratios > ratio*0.7) & (all_aspect_ratios < ratio*1.3))
        if count > len(all_aspect_ratios) * 0.1:  # 超过10%
            common_ratios.append(ratio)
    
    if not common_ratios:
        common_ratios = [1.0, 2.0]
    
    # 为6个层分配宽高比
    aspect_configs = []
    for i in range(6):
        if i < 2:  # 前2层：针孔等小目标
            # 针孔通常接近正方形
            layer_ratios = [r for r in common_ratios if r <= 1.5]
            if not layer_ratios:
                layer_ratios = [1.0]
        elif i < 4:  # 中间层
            layer_ratios = [r for r in common_ratios if 0.7 <= r <= 2.5]
            if not layer_ratios:
                layer_ratios = [1.0, 2.0]
        else:  # 后2层
            layer_ratios = [r for r in common_ratios if r >= 1.0]
            if not layer_ratios:
                layer_ratios = [2.0, 3.0]
        # 转换为Python原生float
        aspect_configs.append([float(r) for r in layer_ratios])
    
    # 5. 计算min_ratio和max_ratio
    min_ratio = float(max(2.0, round(min_sizes[0] * 0.8, 1)))  # 比最小min_size稍小
    max_ratio = 90.0  # 使用float类型
    
    # 6. 打印推荐配置
    print(f"\n推荐的anchor配置:")
    print(f"    aspect_ratios: {aspect_configs}")
    print(f"    min_ratio: {min_ratio}")
    print(f"    max_ratio: {max_ratio}")
    print(f"    min_sizes: {min_sizes}")
    print(f"    max_sizes: {max_sizes}")
    
    # 7. 特别针对针孔的建议
    if 'zhen_kong' in class_stats:
        needle_objs = class_stats['zhen_kong']
        if needle_objs:
            needle_widths = [obj['width'] for obj in needle_objs]
            needle_heights = [obj['height'] for obj in needle_objs]
            avg_size = np.mean([np.sqrt(w*h) for w, h in zip(needle_widths, needle_heights)])
            needle_ratio = (avg_size / model_input_size) * 100
            
            print(f"\n针孔特别分析:")
            print(f"  原始平均尺寸: {np.mean([obj['original_width'] for obj in needle_objs]):.1f}×{np.mean([obj['original_height'] for obj in needle_objs]):.1f} 像素")
            print(f"  缩放后平均尺寸: {np.mean(needle_widths):.1f}×{np.mean(needle_heights):.1f} 像素")
            print(f"  相对于模型输入尺寸: {needle_ratio:.2f}%")
            print(f"  当前第一层min_size: {min_sizes[0]:.1f}%")
            
            if needle_ratio < min_sizes[0] * 0.8:
                print(f"  建议: 针孔尺寸偏小，可考虑降低第一层min_size到 {max(0.5, needle_ratio*1.2):.1f}")
            else:
                print(f"  当前配置可覆盖针孔尺寸")
    
    return {
        'aspect_ratios': aspect_configs,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio,
        'min_sizes': min_sizes,
        'max_sizes': max_sizes
    }

# 主程序
if __name__ == "__main__":
    # 配置参数
    XML_DIR = "./annotations"  # 你的XML文件目录，请修改为实际路径
    MODEL_INPUT_SIZE = 300  # 模型输入尺寸
    
    print("开始分析数据集并计算anchor配置...")
    print("注意: 本程序会考虑图片从原始尺寸缩放到模型输入尺寸的变化")
    config = analyze_dataset_and_calculate_anchors(XML_DIR, MODEL_INPUT_SIZE)
    
    if config:
        print(f"\n" + "="*60)
        print("配置使用说明")
        print("="*60)
        print("将上述配置复制到你的SSD配置文件中，替换对应的anchor_generator部分。")
        print("\n注意事项:")
        print("1. 所有尺寸已根据图片缩放计算，适合模型输入尺寸300×300")
        print("2. 输出值已转换为Python原生float类型，避免np.float64")
        print("3. 如果针孔检测效果仍不理想，可尝试：")
        print("   a. 将第一层min_size降低到1.0-2.0之间")
        print("   b. 针对针孔增加数据增强（随机缩放、对比度增强）")
        print("   c. 调整特征图使用更浅的层：feature_maps: [8, 10, 12, 13, 14, 15]")