from PIL import Image
import os

# 定义图片目录路径和输出图片大小
base_dir = 'results'
output_image = 'results/output/output_32.png'
width, height = 128, 128  # 假设每张图片的大小是 512x512

# 获取子目录列表，选择特定范围的子目录
subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
# 从子目录名称中提取数字，进行范围比较
selected_subdirs = [d for d in subdirs if 'our_inv_' in d and 1724510818 <= int(d.split('_')[-1]) <= 1724511124]


# 定义拼接后的图片大小
total_width = len(selected_subdirs) * width
total_height = 3 * height

# 创建一个空白图片用于拼接
result_image = Image.new('RGB', (total_width, total_height))

# 遍历子目录并按列拼接图片
for i, subdir in enumerate(selected_subdirs):
    subdir_path = os.path.join(base_dir, subdir)
    
    # 获取子目录下的子文件夹
    subdir_path = os.path.join(subdir_path, os.listdir(subdir_path)[0])
    sub_subdirs = os.listdir(subdir_path)

    # 遍历每个子文件夹中的图片文件
    for j, sub_subdir in enumerate(sub_subdirs):
        sub_subdir_path = os.path.join(subdir_path, sub_subdir)
        
        # 获取该子文件夹中的图片文件
        images = [f for f in os.listdir(sub_subdir_path) if os.path.isfile(os.path.join(sub_subdir_path, f))]
        
        image_path = os.path.join(sub_subdir_path, images[0])  # 假设每个子文件夹中只有一张图片
        img = Image.open(image_path)

        # 检查并调整图片大小
        img = img.resize((width, height)) if img.size != (width, height) else img

        # 计算图片粘贴位置
        y_offset = j * height
        x_offset = i * width

        # 检查是否超出边界
        if x_offset + width <= total_width and y_offset + height <= total_height:
            result_image.paste(img, (x_offset, y_offset))
            print(f"粘贴图片 {image_path} 到位置 ({x_offset}, {y_offset})，尺寸为 {width} x {height}")
        else:
            print(f"图片 {image_path} 超出范围，跳过粘贴")

# 检查拼接图像的大小
print(f"拼接图像大小为：{result_image.size}")

# 保存拼接后的图片
result_image.save(output_image)
print(f"拼接完成，图片已保存为 {output_image}")
