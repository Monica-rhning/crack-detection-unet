import os
import torch
import numpy as np
from PIL import Image, ImageFilter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from unet_model import UNet
from wnet import W_Net

def predict_image(model, image_path, device, threshold, output_dir="masks", apply_postprocessing=True):
    """使用训练好的模型对单张图片进行裂缝检测并保存二值掩码"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载图像
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法打开图像 {image_path}: {e}")
        return

    img_name = os.path.basename(image_path)
    base_name = os.path.splitext(img_name)[0]

    # 保存原始尺寸以便后续恢复
    original_size = img.size

    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

    # 应用阈值
    binary_mask = (pred > threshold).astype(np.uint8)

    # 后处理（如果启用）
    if apply_postprocessing:
        # 将NumPy数组转为PIL图像进行处理
        mask_img = Image.fromarray(binary_mask * 255).convert('L')
        # 使用中值滤波去除小噪点
        mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))
        # 转回NumPy数组
        binary_mask = np.array(mask_img) > 204

    # 保存二值化掩码（黑底白裂缝）
    # 裂缝部分为1，乘以255后变为白色(255)，背景为黑色(0)
    binary_mask = binary_mask.astype(np.uint8) * 255
    mask_output = Image.fromarray(binary_mask).resize(original_size, Image.Resampling.LANCZOS)
    mask_path = os.path.join(output_dir, f"{base_name}.png")
    mask_output.save(mask_path)

    print(f"已保存掩码: {mask_path}")
    return mask_path


def process_directory(model, input_dir, device, threshold, output_dir="masks", apply_postprocessing=True):
    """处理目录中所有图像文件"""
    # 支持的图像文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # 检查是否为图像文件
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            print(f"正在处理: {filename}")
            predict_image(
                model,
                file_path,
                device,
                threshold=threshold,
                output_dir=output_dir,
                apply_postprocessing=apply_postprocessing
            )

    print("所有图像处理完成")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用训练好的模型对文件夹中的图像进行裂缝检测')
    parser.add_argument('--input_dir', type=str, default='UAV-Crack-dataset/leftImg8bit/val', help='输入图像文件夹路径')
    parser.add_argument('--model', type=str, default='output_results/HWDU_best_model.pth', help='模型权重文件')
    parser.add_argument('--output_dir', type=str, default='HWDU_val_masks', help='掩码输出目录')
    parser.add_argument('--threshold', type=float, default=0.9, help='二值化阈值，范围0-1，默认0.5')
    parser.add_argument('--no-postprocessing', action='store_true', help='禁用后处理操作')
    args = parser.parse_args()

    # 检查输入文件夹和模型文件是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入文件夹不存在: {args.input_dir}")
        return

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    # model = W_Net()
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print(f"已加载模型: {args.model}")

    # 处理目录中所有图像
    process_directory(
        model,
        args.input_dir,
        device,
        threshold=args.threshold,
        output_dir=args.output_dir,
        apply_postprocessing=args.no_postprocessing
    )


if __name__ == '__main__':
    main()