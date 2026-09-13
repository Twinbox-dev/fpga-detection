import os
import shutil
import random

def create_test_dataset():
    """
    从数据集中随机选择图片创建测试集
    每个类别的数量在代码中固定
    """
    # 原始数据目录
    data_dir = "./"  # 当前目录
    
    # 测试集目录
    test_dir = "./test"
    
    # 设置随机种子（保证结果可复现）
    random.seed(42)
    
    # 每个类别需要的图片数量
    category_counts = {
        "正常": 5,
        "脏污": 5, 
        "针孔": 5,
        "擦伤": 5,
        "褶皱": 5
    }
    
    # 类别名与文件名关键字的映射
    category_keywords = {
        "正常": "zhengchang",
        "脏污": "zang_wu", 
        "针孔": "zhen_kong",
        "擦伤": "ca_shang",
        "褶皱": "zhe_zhou"
    }
    
    # 确保测试目录存在
    test_images = os.path.join(test_dir, "images")
    test_annotations = os.path.join(test_dir, "annotations")
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_annotations, exist_ok=True)
    
    # 原始数据目录
    data_images = os.path.join(data_dir, "images")
    data_annotations = os.path.join(data_dir, "annotations")
    
    if not os.path.exists(data_images) or not os.path.exists(data_annotations):
        print("错误: 请确保当前目录有images和annotations文件夹")
        return
    
    # 按类别收集文件（通过文件名中的关键字）
    category_files = {}
    
    # 遍历所有XML文件，按文件名中的关键字分类
    for xml_name in os.listdir(data_annotations):
        if not xml_name.endswith('.xml'):
            continue
            
        xml_path = os.path.join(data_annotations, xml_name)
        base_name = xml_name.replace('.xml', '')
        img_name = base_name + '.jpg'
        img_path = os.path.join(data_images, img_name)
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            continue
        
        # 根据文件名中的关键字确定类别
        for category, keyword in category_keywords.items():
            if keyword in base_name:
                if category not in category_files:
                    category_files[category] = []
                category_files[category].append(base_name)
                break
        else:
            # 如果没有匹配到任何关键字，尝试从XML中读取类别
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<name>' in content and '</name>' in content:
                    start = content.find('<name>') + 6
                    end = content.find('</name>')
                    xml_category = content[start:end]
                    
                    if xml_category in category_counts:
                        if xml_category not in category_files:
                            category_files[xml_category] = []
                        category_files[xml_category].append(base_name)
                    else:
                        print(f"跳过: {base_name} (类别 '{xml_category}' 不在配置中)")
    
    print(f"按类别统计结果:")
    for category, files in category_files.items():
        print(f"类别 '{category}': {len(files)} 张图片")
    
    # 从每个类别中随机选择图片
    selected_files = []
    
    for category, need_count in category_counts.items():
        if category not in category_files:
            print(f"警告: 数据集没有类别 '{category}'")
            continue
            
        available_files = category_files[category]
        actual_count = min(need_count, len(available_files))
        
        if actual_count < need_count:
            print(f"警告: 类别 '{category}' 只有 {len(available_files)} 张，但需要 {need_count} 张")
        
        # 随机选择
        selected = random.sample(available_files, actual_count)
        print(f"从 '{category}' 中选择了 {actual_count} 张")
        
        for base_name in selected:
            selected_files.append(base_name)
    
    # 复制文件到测试集
    copied_count = 0
    for base_name in selected_files:
        # 源文件
        src_img = os.path.join(data_images, f"{base_name}.jpg")
        src_xml = os.path.join(data_annotations, f"{base_name}.xml")
        
        # 目标文件
        dst_img = os.path.join(test_images, f"{base_name}.jpg")
        dst_xml = os.path.join(test_annotations, f"{base_name}.xml")
        
        # 复制文件
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_xml, dst_xml)
        copied_count += 1
    
    print(f"\n完成! 已将 {copied_count} 张图片复制到 {test_dir}")
    print(f"图片: {test_images}")
    print(f"标注: {test_annotations}")

if __name__ == "__main__":
    create_test_dataset()