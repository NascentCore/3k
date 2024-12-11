# 项目介绍
本项目是一个demo演示项目，目的是展示一个图片检索功能。

## 图片文件
大约300M的图片文件，内容都是飞机的图片。

## 模块
- 图片预处理模块
- 图片检索模块

### 图片预处理模块
preprocess.py会遍历目标目录下的所有图片文件，并依次执行以下操作：
- 调用多模态大模型接口，提取图片的航空公司和飞机型号
- 将图片路径、航空公司、飞机型号写入sqlite数据库

提示词
```
你是一个图片识别专家，请根据图片内容，给出图片中飞机的航空公司，一般在机身上有航空公司的logo或文字。
可能的航空公司有：
- ANA全日空航空(ANA - All Nippon Airways    )
- 汉莎航空(Lufthansa)
- 星空联盟(Star Alliance)
- 国际航空(IA - Air China)
- 圆通航空(YTO - Yunda Express)
- 天骄航空(TJU - Tianjiao Airlines)
- 东方航空(CEA - China Eastern Airlines)
- 南方航空(CSN - China Southern Airlines)
- 成都航空(CTU - Chengdu Airlines)
- 一二三航空(UNI - Uni Air)
```

### 图片检索模块
- ptyhon web服务
    - 可以上传图片，并检索出相似的图片
    - 也支持文本检索
- 前端
    - 使用vue3框架，展示图片检索结果
    - 使用element-plus框架，展示图片检索结果