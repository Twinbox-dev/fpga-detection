import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_anchors_for_layer(layer_idx, min_size, aspect_ratios, grid_size=10):
    """可视化某一层的Anchor分布"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    
    # 计算网格步长
    stride = 300 / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            cx = (i + 0.5) * stride
            cy = (j + 0.5) * stride
            
            # 绘制正方形Anchor
            w = h = min_size
            rect = patches.Rectangle((cx-w/2, cy-h/2), w, h,
                                   linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # 绘制各比例Anchor
            for ar in aspect_ratios:
                h_ar = min_size / (ar**0.5)
                w_ar = h_ar * ar
                rect = patches.Rectangle((cx-w_ar/2, cy-h_ar/2), w_ar, h_ar,
                                       linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
    
    ax.set_title(f'Layer {layer_idx}: Anchors (min_size={min_size})')
    plt.show()

# 可视化第1层
visualize_anchors_for_layer(1, 4.2, [0.7], grid_size=38)