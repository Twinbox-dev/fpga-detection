import os
import xml.etree.ElementTree as ET
from pathlib import Path

def set_difficult_for_all_objects(xml_folder_path, difficult_value=1, file_extension='.xml'):
    """
    为指定文件夹内所有XML文件中所有object标签下的difficult子标签设置统一值。

    参数:
        xml_folder_path (str): 存放XML文件的文件夹路径。
        difficult_value (int): 要设置的difficult标签值，默认为1。
        file_extension (str): XML文件的扩展名，默认为'.xml'。
    """
    # 将输入路径转换为Path对象以便处理
    folder_path = Path(xml_folder_path)
    
    # 检查文件夹是否存在
    if not folder_path.exists():
        print(f"错误：文件夹 '{xml_folder_path}' 不存在。")
        return
    
    # 遍历文件夹，查找所有扩展名为.xml的文件
    xml_files = list(folder_path.glob(f"*{file_extension}"))
    
    if not xml_files:
        print(f"在文件夹 '{xml_folder_path}' 中未找到{file_extension}文件。")
        return
    
    processed_count = 0
    
    for xml_file in xml_files:
        try:
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 查找所有object标签
            objects = root.findall('.//object')
            
            if not objects:
                print(f"提示：文件 '{xml_file.name}' 中未找到object标签。")
                continue
            
            # 遍历所有object标签，修改其difficult子标签
            for obj in objects:
                difficult_elem = obj.find('difficult')
                if difficult_elem is not None:
                    # 修改difficult标签的文本内容
                    difficult_elem.text = str(difficult_value)
                else:
                    # 如果该object标签下没有difficult标签，则创建一个新的
                    new_difficult = ET.SubElement(obj, 'difficult')
                    new_difficult.text = str(difficult_value)
            
            # 保存修改后的XML文件（覆盖原文件）
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            
            print(f"已处理文件: {xml_file.name}，修改了 {len(objects)} 个object标签。")
            processed_count += 1
            
        except ET.ParseError as e:
            print(f"解析文件 '{xml_file.name}' 时出错: {e}")
        except Exception as e:
            print(f"处理文件 '{xml_file.name}' 时发生未知错误: {e}")
    
    print(f"\n处理完成！成功处理 {processed_count} 个XML文件。")

# 使用示例
if __name__ == "__main__":
    xml_folder = "./difficult"  
    
    # 调用函数，将所有difficult标签设置为1
    set_difficult_for_all_objects(xml_folder, difficult_value=1)