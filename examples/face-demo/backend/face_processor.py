import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from config import (DEVICE)
from util import logger

class FaceProcessor:
    def __init__(self):
        logger.info("初始化 FaceProcessor...")
        # 使用配置中的设备设置
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        logger.info("加载 MTCNN 模型...")
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            select_largest=False,
            post_process=True,
            image_size=160
        )
        
        logger.info("加载 FaceNet 模型...")
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def extract_face(self, image_path):
        """
        从图片中提取人脸信息
        Args:
            image_path: 图片路径
        Returns:
            tuple: (success, result)
                - success: bool, 表示处理是否成功
                - result: 成功时返回包含人脸信息的字典，失败时返回错误信息
                    成功时的字典格式：{
                        'embeddings': np.ndarray,  # 人脸特征向量数组
                        'boxes': np.ndarray,       # 人脸框位置数组
                        'probs': np.ndarray,       # 置信度数组
                        'face_count': int          # 检测到的人脸数量
                    }
        """
        try:
            logger.info(f"\n开始处理图片: {image_path}")
            
            # 读取图片
            logger.info("读取图片...")
            img = Image.open(image_path)
            if img.mode != 'RGB':
                logger.info("转换图片为 RGB 模式")
                img = img.convert('RGB')
                
            # 检测人脸和提取特征
            logger.info("检测人脸...")
            # 使用 MTCNN 检测人脸
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None or not isinstance(boxes, np.ndarray):
                msg = f"在图片 {image_path} 中未检测到有效的人脸"
                logger.warning(msg)
                return False, msg
            
            logger.info(f"检测到 {len(boxes)} 个人脸")
                
            # 对每个检测到的人脸进行处理
            face_tensors = []
            valid_boxes = []
            valid_probs = []
            logger.info("提取人脸区域...")
            
            for i, (box, prob) in enumerate(zip(boxes, probs), 1):
                try:
                    # 检查置信度
                    if prob < 0.9:
                        logger.warning(f"警告: 第 {i} 个人脸的置信度太低 ({prob:.2f})，跳过")
                        continue
                    
                    # 确保边界框坐标是整数且有效
                    x1, y1, x2, y2 = map(int, box)
                    if x1 < 0 or y1 < 0 or x2 > img.width or y2 > img.height:
                        logger.warning(f"警告: 第 {i} 个人脸框超出图片范围，跳过")
                        continue
                    
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        logger.warning(f"警告: 第 {i} 个人脸太小，跳过")
                        continue
                    
                    # 提取人脸
                    face = self.mtcnn.extract(img, [[x1, y1, x2, y2]], save_path=None)
                    if face is not None and isinstance(face, torch.Tensor):
                        # 校验维度是否为 [3, H, W]
                        if len(face.shape) == 4:
                            face = face.squeeze(0)  # 如果是批量的形状 [1, 3, H, W]，缩到 [3, H, W]
                        elif len(face.shape) != 3:
                            logger.warning(f"警告: 提取到的人脸形状异常 {face.shape}，跳过")
                            continue
                        face_tensors.append(face)
                        valid_boxes.append(box)
                        valid_probs.append(prob)
                        logger.info(f"成功提取第 {i} 个人脸")
                    else:
                        logger.warning(f"警告: 第 {i} 个人脸提取失败")
                        
                except Exception as extract_err:
                    logger.warning(f"警告: 提取第 {i} 个人脸时出错: {str(extract_err)}")
                    continue
                    
            if not face_tensors:
                msg = "没有成功提取到任何人脸，跳过该图片"
                logger.info(msg)
                return False, msg
                
            # 批量提取特征向量
            logger.info("提取特征向量...")
            try:
                face_tensors = torch.stack(face_tensors).to(self.device)
                embeddings = self.facenet(face_tensors).detach().cpu().numpy()
                
                result = {
                    'embeddings': embeddings,
                    'boxes': np.array(valid_boxes),
                    'probs': np.array(valid_probs),
                    'face_count': len(embeddings)
                }
                
                logger.info(f"✓ 成功处理图片 {image_path}，共提取 {len(embeddings)} 个人脸")
                return True, result
                
            except Exception as embed_err:
                logger.error(f"特征提取失败: {str(embed_err)}")
                return False, f"特征提取失败: {str(embed_err)}"
                
        except Exception as e:
            error_msg = f"处理图片 {image_path} 时出错: {str(e)}"
            logger.error(error_msg)
            logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            return False, error_msg
