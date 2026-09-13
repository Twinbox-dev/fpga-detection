import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 真实框
gt = [400, 180, 420, 199]  # x1, y1, x2, y2
pred = [408, 176, 424, 194]  # x1, y1, x2, y2

# 提取坐标
gt_x1, gt_y1, gt_x2, gt_y2 = gt
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
inter_width = inter_x2 - inter_x1
inter_height = inter_y2 - inter_y1

# 计算面积
gt_area = gt_width * gt_height
pred_area = pred_width * pred_height
inter_area = inter_width * inter_height
union_area = gt_area + pred_area - inter_area
iou = inter_area / union_area

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# 反转Y轴（图片坐标系）
ax.invert_yaxis()

# 画真实框（红色）
gt_rect = patches.Rectangle(
    (gt_x1, gt_y1), gt_width, gt_height,
    linewidth=2, edgecolor='red', facecolor='none', alpha=0.8, label='Ground Truth'
)
ax.add_patch(gt_rect)

# 画预测框（蓝色）
pred_rect = patches.Rectangle(
    (pred_x1, pred_y1), pred_width, pred_height,
    linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8, label='Prediction'
)
ax.add_patch(pred_rect)

# 画交集区域（绿色）
if inter_width > 0 and inter_height > 0:
    inter_rect = patches.Rectangle(
        (inter_x1, inter_y1), inter_width, inter_height,
        linewidth=1, edgecolor='green', facecolor='green', alpha=0.3, label='Intersection'
    )
    ax.add_patch(inter_rect)

# 标记框的坐标
# 真实框坐标
ax.text(gt_x1, gt_y1-2, f'({gt_x1},{gt_y1})', fontsize=8, ha='center', color='red')
ax.text(gt_x2, gt_y2+2, f'({gt_x2},{gt_y2})', fontsize=8, ha='center', color='red')

# 预测框坐标
ax.text(pred_x1, pred_y1+2, f'({pred_x1},{pred_y1})', fontsize=8, ha='center', color='blue')
ax.text(pred_x2, pred_y2-2, f'({pred_x2},{pred_y2})', fontsize=8, ha='center', color='blue')

# 设置坐标范围
padding = 10
x_min = min(gt_x1, pred_x1) - padding
x_max = max(gt_x2, pred_x2) + padding
y_min = min(gt_y1, pred_y1) - padding
y_max = max(gt_y2, pred_y2) + padding

# 设置坐标轴
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_max, y_min)  # Y轴向下增加
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 添加标题
ax.set_title(f'Bounding Box IOU: {iou:.3f}')

# 显示IOU计算信息
info_text = f'IOU = {inter_area} / {union_area} = {iou:.3f}\n' \
            f'GT: {gt_width}x{gt_height} = {gt_area}\n' \
            f'Pred: {pred_width}x{pred_height} = {pred_area}\n' \
            f'Inter: {inter_width}x{inter_height} = {inter_area}'

ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 添加图例
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 输出IOU值
print(f"IOU: {iou:.3f}")