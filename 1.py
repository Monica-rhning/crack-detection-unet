import cv2
import numpy as np
from PIL import Image


def crack_detection_physical(image_path, output_path="crack_binary.png",
                             blur_kernel=(5, 5), canny_low=50, canny_high=150,
                             morph_kernel=(3, 3), iterations=1):
    """
    基于物理/传统图像处理的裂缝识别，生成黑底白裂缝的二值图像

    参数：
        image_path: 输入图片路径
        output_path: 二值裂缝图像保存路径
        blur_kernel: 高斯模糊核大小（去噪用）
        canny_low/canny_high: Canny边缘检测的高低阈值
        morph_kernel: 形态学操作核大小（优化裂缝连接性）
        iterations: 形态学操作迭代次数
    """
    # 1. 读取图像并转为灰度图（简化处理，聚焦亮度差异）
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊去噪（减少噪声对边缘检测的干扰）
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 3. Canny边缘检测（提取裂缝的边缘特征，利用裂缝与背景的亮度差）
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # 4. 形态学操作（膨胀连接断裂裂缝，腐蚀去除小噪声）
    kernel = np.ones(morph_kernel, np.uint8)
    # 先膨胀：连接裂缝的断裂部分
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    # 后腐蚀：去除膨胀带来的噪声点，细化裂缝
    eroded = cv2.erode(dilated, kernel, iterations=iterations)

    # 5. 生成二值图像（黑底白裂缝）
    binary = np.where(eroded > 0, 255, 0).astype(np.uint8)

    # 6. 保存结果
    cv2.imwrite(output_path, binary)
    print(f"裂缝二值图像已保存至：{output_path}")

    # 返回二值图像（可选）
    return Image.fromarray(binary)


# -------------------------- 使用示例 --------------------------
if __name__ == "__main__":
    # 替换为你的图片路径（支持jpg/png等格式）
    input_image = r"D:\drop_train\CHANGAN_AI\crack-detection-unet-master\crack-detection-unet-master\UAV-Crack-dataset\leftImg8bit\train\UAV-CrackX16\937958_DJI_20231015160920_0117_Z.JPG_5_z.jpg"
    # 调用函数识别裂缝
    crack_detection_physical(
        image_path=input_image,
        output_path="crack_binary_result.png",
        blur_kernel=(5, 5),  # 噪声多可增大核（如(7,7)）
        canny_low=40, canny_high=120,  # 裂缝模糊可降低low阈值
        morph_kernel=(3, 3), iterations=2  # 裂缝断裂严重可增加iterations
    )