import os
import sys
import re
import cv2
import torch.nn.functional as F
from paddleocr import PaddleOCR, draw_ocr
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # 查看可用的设备数量
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA Devices: {num_devices}")
    
    # 列出每个设备的信息
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
        
torch.cuda.set_device(5)
current_device = torch.cuda.current_device()
print(f"Current Device: {current_device}")

device = torch.device("cuda:5")

# 初始化matting模型
matting_model = models.segmentation.deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')
matting_model = matting_model.to(device).eval()

## 初始化OCR模型
#ocr_ch = PaddleOCR(use_angle_cls=True, lang='ch')  # lang='ch'支持中文
#ocr_en = PaddleOCR(use_angle_cls=True, lang='en')  # 英文识别

# 初始化OCR模型
ocr_ch = PaddleOCR(use_angle_cls=True,  # 启用文字方向分类模块
        det_db_box_thresh=0.1,  # 调低检测框阈值
        det_db_unclip_ratio=5.0,  # 增大检测框的扩展比例
        lang='ch',  # lang='ch'支持中文
        use_gpu=True)

ocr_en = PaddleOCR(use_angle_cls=True,  # 启用文字方向分类模块
        det_db_box_thresh=0.1,  # 调低检测框阈值
        det_db_unclip_ratio=5.0,  # 增大检测框的扩展比例
        lang='en',  # 英文识别
        use_gpu=True)

#def resize_image(input_image, target_size=(520, 520)):
#    h, w, _ = input_image.shape
#    target_h, target_w = target_size
#
#    # 计算缩放比例
#    scale = min(target_w / w, target_h / h)
#    
#    # 按比例缩放图像
#    new_w = int(w * scale)
#    new_h = int(h * scale)
#    resized_image = cv2.resize(input_image, (new_w, new_h))
#
#    # 将图像填充到目标尺寸
#    top = (target_h - new_h) // 2
#    bottom = target_h - new_h - top
#    left = (target_w - new_w) // 2
#    right = target_w - new_w - left
#
#    # 填充图像，使其匹配目标尺寸
#    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#
#    return padded_image

def resize_image(input_image, target_size=(520, 520)):
    h, w, _ = input_image.shape
    target_h, target_w = target_size

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)

    # 按比例缩放图像
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(input_image, (new_w, new_h))

    # 将图像填充到目标尺寸
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    # 填充图像，使其匹配目标尺寸
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image, (h, w), (new_h, new_w)  # 返回原始尺寸和缩放后的尺寸

def adjust_threshold_for_airplane(output, threshold_factor=2.0):
    background_class = 0
    airplane_class = 1
    background_scores = output[background_class, :, :]
    airplane_scores = output[airplane_class, :, :]
    
    # 扩大飞机类别得分
    adjusted_airplane_scores = airplane_scores + threshold_factor
    
    # 比较调整后的飞机得分和背景得分，选择更高的得分
    output_with_adjusted_airplane = (adjusted_airplane_scores > background_scores).float()
    
    # 使用 argmax 获取最终的类别预测
    final_prediction = torch.argmax(output, dim=0)  # 默认沿着类别维度选择最大得分类别

    # 将调整后的飞机预测区域强制设置为飞机类别
    final_prediction[output_with_adjusted_airplane == 1] = airplane_class
    airplane_percentage = (final_prediction == 1).sum().item() / final_prediction.numel()
    print(f"飞机类别占比: {airplane_percentage * 100:.2f}%")
    
    return final_prediction

def filter_ocr_by_background_with_threshold(ocr_results, background_mask, threshold=0.9):
    """
    根据背景掩码和阈值比例过滤 OCR 结果。
    如果一个文本框的背景像素占比高于阈值，则丢弃。
    
    :param ocr_results: OCR 结果列表，格式为 [(lang, text, score, bbox), ...]，
                        bbox 是文本框的四个角坐标 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]。
    :param background_mask: 背景掩码 (tensor)，形状为 [H, W]，每个像素 True 表示背景。
    :param threshold: 背景像素比例阈值，超过该比例的文本框将被过滤。
    :return: 筛选后的 OCR 结果列表。
    """
    filtered_results = []
    height, width = background_mask.shape  # 背景掩码的尺寸

    for result in ocr_results:
        lang, text, score, bbox = result
        
        # 将 bbox 转换为 numpy 格式，并限制坐标在有效范围内
        polygon = np.array([[int(x), int(y)] for x, y in bbox], dtype=np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        
        # 创建一个与背景掩码同大小的临时掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 在临时掩码上填充多边形区域
        cv2.fillPoly(mask, [polygon], 1)  # 多边形区域标记为 1
        
        # 将 mask 转换为 torch.Tensor 并移动到与 background_mask 相同的设备
        mask_tensor = torch.tensor(mask, device=background_mask.device, dtype=torch.bool)

        # 计算文本区域内像素的总数和背景像素数量
        total_pixels = mask_tensor.sum().item()  # 文本框内总像素数
        background_pixels = (background_mask & mask_tensor).sum().item()  # 文本框内背景像素数

        # 计算背景像素占比
        if total_pixels > 0:  # 避免除以 0
            background_ratio = background_pixels / total_pixels
        else:
            background_ratio = 1.0  # 如果区域为空，默认为背景区域

        # 根据比例阈值过滤文本框
        if background_ratio <= threshold:  # 背景比例小于等于阈值，保留
            filtered_results.append(result)
    
    return filtered_results


def shear_image(image, shear_factor=0.2):
    """
    对图片进行水平 Shear 变换
    :param image: 输入图像 (numpy array)
    :param shear_factor: Shear 变换系数
    :return: 变换后的图像
    """
    rows, cols, channels = image.shape
    # 定义 Shear 矩阵
    shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    # 计算新的宽度
    new_width = int(cols + abs(shear_factor) * rows)
    sheared_image = cv2.warpAffine(image, shear_matrix, (new_width, rows), borderMode=cv2.BORDER_REPLICATE)
    return sheared_image

def rotate_image(image, angle):
    """
    旋转图像
    :param image: 输入图像 (numpy array)
    :param angle: 旋转角度 (正值为逆时针，负值为顺时针)
    :return: 旋转后的图像
    """
    height, width = image.shape[:2]  # 获取图像的高度和宽度
    center = (width // 2, height // 2)  # 计算旋转中心点

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的边界尺寸
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵，考虑平移
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # 应用仿射变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_REPLICATE)

    return rotated_image

def calculate_effective_length(text):
    """
    根据规则计算字符串的有效长度：
    - 每个汉字算2个字
    - 非汉字字符算1个字
    :param text: 输入字符串
    :return: 有效长度
    """
    effective_length = 0
    for char in text:
        if re.match(r"[\u4e00-\u9fff]", char):  # 汉字的 Unicode 范围
            effective_length += 2
        else:
            effective_length += 1
    return effective_length

def ocr_paddle(image, ocr_ch, ocr_en):
    """
    对图像进行OCR，支持中文和英文
    :param image: 输入图像 (numpy array, RGB格式)
    :param ocr_ch: 中文OCR模型
    :param ocr_en: 英文OCR模型
    :return: OCR结果
    """
    result_ch = ocr_ch.ocr(image, rec=True, cls=True)
    result_en = ocr_en.ocr(image, rec=True, cls=True)

    all_results = []
    for line in result_ch:
        if line is None:
            continue
        for box in line:
            text = box[1][0]  # 识别的文本
            score = box[1][1]  # 置信度
            bbox = box[0]      # 文本框坐标
            all_results.append(('ch', text, score, bbox))
    for line in result_en:
        if line is None:
            continue
        for box in line:
            text = box[1][0]  # 识别的文本
            score = box[1][1]  # 置信度
            bbox = box[0]      # 文本框坐标
            all_results.append(('en', text, score, bbox))
    return all_results

def matting_airplance(input_image):
    print(input_image.shape)
    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    #resized_image = resize_image(input_image)
    resized_image, original_size, resized_size = resize_image(input_image)
    input_tensor = preprocess(resized_image).unsqueeze(0)  # 增加批次维度
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = matting_model(input_tensor)['out'][0]  # 获取模型输出

    # 对 output 进行插值，恢复到原始尺寸
    matting_result_origin_size = F.interpolate(output.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
    predicted_classes = torch.argmax(matting_result_origin_size, dim=0)
    background_mask = predicted_classes == 0
    return background_mask

def multi_ocr(input_path):
    """
    构建需要进行OCR的图片列表，并逐一处理
    :param input_path: 图像路径
    :return: 综合OCR结果
    """
    # 使用 OpenCV 读取图像
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"无法读取图片，请检查路径是否正确：{input_path}")
    
    # 转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建待处理图像列表
    processed_images = [image]  # 包含原始图像
    for shear_factor in [0.2, 0.5, 0.7, -0.5, -0.2, -0.7]:
        sheared_image = shear_image(image, shear_factor)  # 生成水平 Shear 图像
        processed_images.append(sheared_image)
    # 添加旋转图像
    for angle in [15, 30, 45, 60, -15, -30, -45, -60]:  # 旋转角度：左旋45度，右旋45度
        rotated_image = rotate_image(image, angle)
        processed_images.append(rotated_image)

    # 循环对每张图像进行OCR
    all_results = []
    for processed_image in processed_images:
        background_mask = matting_airplance(processed_image)
        ocr_results = ocr_paddle(processed_image, ocr_ch, ocr_en)
        filtered_ocr_results = filter_ocr_by_background_with_threshold(ocr_results, background_mask, threshold=0.9)
        for lang, text, score, _ in filtered_ocr_results:  # 忽略文本框
            all_results.append((lang, text.replace(' ', ''), score))

    # 合并结果，按文本去重（保留最高分的结果）
    unique_results = {}
    for lang, text, score in all_results:
        if text not in unique_results or score > unique_results[text][1]:  # 如果分值更高，更新
            unique_results[text] = (lang, score)
    
    # 转换为排序列表，按分值降序排列
    final_results = sorted([(text, lang, score) for text, (lang, score) in unique_results.items()], key=lambda x: x[2], reverse=True)
    # 进一步过滤结果
    final_results = [ (text, score) for text, _,  score in final_results if calculate_effective_length(text) >= 3 ]
    final_results = [ (text, score) for text, score in final_results if score > 0.6]

    return final_results


def main():
    #image_path = '../pics/input/C919_comac (7).jpg'
    input_dir = '../pics/input'
    for filename in os.listdir(input_dir):
        #filename = 'C919_internet(43).jpeg'
        #if ('C919' not in filename):
        #if (filename != 'C919_comac (6).jpg'):
        #    continue
        input_file_path = os.path.join(input_dir, filename)
        ocr_results = multi_ocr(input_file_path)
        #sorted_ocr_results = sorted(ocr_results, key=lambda x: x[2], reverse=True)
        str_ocr_results = "  ".join([f"{text}:{score}" for text, score in ocr_results])
        print (filename + "  " +str_ocr_results , file=sys.stderr)
        #break
        #print (f"\n{filename}", file=sys.stderr)
        #for ocr_result in ocr_results:
        #    print (f"{ocr_result[0]}  {ocr_result[1]}  {ocr_result[2]}", file=sys.stderr)
 


if __name__ == '__main__':
    #torch.set_printoptions(precision=2, sci_mode=False, linewidth=220, profile="full")
    main()
