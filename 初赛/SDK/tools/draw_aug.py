import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 命令行美化
def print_separator():
    print("=" * 60)

def print_header():
    print_separator()
    print(" " * 20 + "边框交并比(IoU)计算器")
    print_separator()

def print_menu():
    print("\n请输入边框坐标:")
    print("格式: x1 y1 x2 y2")
    print("(x1,y1) = 左上角坐标")
    print("(x2,y2) = 右下角坐标")
    print("(注意: 使用图片坐标系: (0,0) 在左上角)")
    print("-" * 40)

# 输入函数
def input_box(prompt):
    while True:
        try:
            coords = input(prompt)
            values = list(map(int, coords.split()))
            if len(values) != 4:
                print("错误: 请输入4个用空格分隔的数字")
                continue
            x1, y1, x2, y2 = values
            if x1 >= x2 or y1 >= y2:
                print("错误: x2 必须大于 x1, 且 y2 必须大于 y1")
                continue
            return [x1, y1, x2, y2]
        except ValueError:
            print("错误: 请输入有效的整数")
        except KeyboardInterrupt:
            print("\n\n程序被用户终止。")
            exit()

# 主程序
def main():
    # 清屏
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 打印标题
    print_header()
    print_menu()
    
    # 输入真实框
    print("\n1. 真实框 (Ground Truth)")
    print("请输入真实框坐标:")
    gt = input_box(">>> ")
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    
    # 输入预测框
    print("\n2. 预测框 (Prediction)")
    print("请输入预测框坐标:")
    pred = input_box(">>> ")
    pred_x1, pred_y1, pred_x2, pred_y2 = pred
    
    # 计算交集
    inter_x1 = max(gt_x1, pred_x1)
    inter_y1 = max(gt_y1, pred_y1)
    inter_x2 = min(gt_x2, pred_x2)
    inter_y2 = min(gt_y2, pred_y2)
    
    # 计算尺寸
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    pred_width = pred_x2 - pred_x1
    pred_height = pred_y2 - pred_y1
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    
    # 计算面积
    gt_area = gt_width * gt_height
    pred_area = pred_width * pred_height
    inter_area = inter_width * inter_height
    union_area = gt_area + pred_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    # 显示输入信息
    print_separator()
    print("\n" + " " * 20 + "输入信息汇总")
    print_separator()
    print(f"真实框: ({gt_x1}, {gt_y1}) → ({gt_x2}, {gt_y2})")
    print(f"        宽度: {gt_width}, 高度: {gt_height}, 面积: {gt_area}")
    print(f"预测框: ({pred_x1}, {pred_y1}) → ({pred_x2}, {pred_y2})")
    print(f"        宽度: {pred_width}, 高度: {pred_height}, 面积: {pred_area}")
    
    # 显示计算结果
    print_separator()
    print("\n" + " " * 20 + "计算结果")
    print_separator()
    print(f"交集区域: ({inter_x1}, {inter_y1}) → ({inter_x2}, {inter_y2})")
    print(f"          宽度: {inter_width}, 高度: {inter_height}, 面积: {inter_area}")
    print(f"并集面积: {union_area}")
    print(f"交并比(IoU): {inter_area} / {union_area} = {iou:.4f}")
    
    if inter_width <= 0 or inter_height <= 0:
        print("\n⚠️  警告: 两个边框没有交集!")
    else:
        print(f"\n✓ 找到交集区域。")
    
    # 询问是否绘图
    print_separator()
    plot_option = input("\n是否要可视化边框? (y/n): ").lower()
    
    if plot_option == 'y':
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 反转Y轴（图片坐标系）
        ax.invert_yaxis()
        
        # 画真实框（红色） - 保持英文标签
        gt_rect = patches.Rectangle(
            (gt_x1, gt_y1), gt_width, gt_height,
            linewidth=2.5, edgecolor='red', facecolor='none', alpha=0.9, label='Ground Truth'
        )
        ax.add_patch(gt_rect)
        
        # 画预测框（蓝色） - 保持英文标签
        pred_rect = patches.Rectangle(
            (pred_x1, pred_y1), pred_width, pred_height,
            linewidth=2.5, edgecolor='blue', facecolor='none', alpha=0.9, label='Prediction'
        )
        ax.add_patch(pred_rect)
        
        # 画交集区域（如果有交集） - 保持英文标签
        if inter_width > 0 and inter_height > 0:
            inter_rect = patches.Rectangle(
                (inter_x1, inter_y1), inter_width, inter_height,
                linewidth=2, edgecolor='green', facecolor='green', alpha=0.3, label='Intersection'
            )
            ax.add_patch(inter_rect)
        
        # 标记框的坐标
        # 真实框坐标
        ax.text(gt_x1, gt_y1-3, f'GT: ({gt_x1},{gt_y1})', fontsize=9, ha='center', 
                color='red', weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        ax.text(gt_x2, gt_y2+3, f'({gt_x2},{gt_y2})', fontsize=8, ha='center', 
                color='red', bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
        
        # 预测框坐标
        ax.text(pred_x1, pred_y1+3, f'Pred: ({pred_x1},{pred_y1})', fontsize=9, ha='center', 
                color='blue', weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        ax.text(pred_x2, pred_y2-3, f'({pred_x2},{pred_y2})', fontsize=8, ha='center', 
                color='blue', bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
        
        # 设置坐标范围
        padding = 20
        x_min = min(gt_x1, pred_x1, inter_x1) - padding
        x_max = max(gt_x2, pred_x2, inter_x2) + padding
        y_min = min(gt_y1, pred_y1, inter_y1) - padding
        y_max = max(gt_y2, pred_y2, inter_y2) + padding
        
        # 设置坐标轴
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Y轴向下增加
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('X (→ increasing)', fontsize=12)
        ax.set_ylabel('Y (← increasing)', fontsize=12)
        
        # 添加标题 - 保持英文
        title_color = 'green' if iou > 0.5 else 'orange' if iou > 0.3 else 'red'
        ax.set_title(f'Bounding Box Visualization - IoU = {iou:.4f}', 
                    fontsize=14, fontweight='bold', color=title_color, pad=20)
        
        # 显示IOU计算信息 - 保持英文
        info_text = f'IoU = {inter_area} / {union_area} = {iou:.4f}\n' \
                   f'GT Area: {gt_width}×{gt_height} = {gt_area}\n' \
                   f'Pred Area: {pred_width}×{pred_height} = {pred_area}\n' \
                   f'Intersection: {inter_width}×{inter_height} = {inter_area}'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # 添加坐标轴方向指示 - 保持英文
        ax.annotate('→ X increases', xy=(0.95, 0.05), xycoords='axes fraction',
                   fontsize=9, ha='right', color='gray')
        ax.annotate('↓ Y increases', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, va='top', color='gray')
        
        # 添加图例 - 保持英文
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        print("\n🎯 正在生成可视化图表...")
        plt.show()
        
        # 保存选项
        save_option = input("\n是否保存图表? (y/n): ").lower()
        if save_option == 'y':
            filename = input("请输入文件名(不含扩展名): ").strip()
            if filename:
                filename = f"{filename}.png" if '.' not in filename else filename
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✅ 图表已保存为 '{filename}'")
    
    print_separator()
    print("\n" + " " * 20 + "程序完成")
    print_separator()
    
    # 是否继续
    continue_option = input("\n是否计算另一个IoU? (y/n): ").lower()
    if continue_option == 'y':
        main()
    else:
        print("\n感谢使用 IoU 计算器!")
        print_separator()

# 运行程序
if __name__ == "__main__":
    main()