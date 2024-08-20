from PIL import Image
import os

# 定义图片路径
image_path = "results/our_inv_1723732794/src_a_photo_of_a_smiling_man/dec_a_photo_of_a_serious_man/cfg_d_15.0_skip_36_1723732851.png"
# 打开图片并获取大小
img = Image.open(image_path)
width, height = img.size

# 输出图片的宽度和高度
print(f"图片大小为：{width} x {height}")
