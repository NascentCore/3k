# 导入本地文档至知识库

## 导入步骤
1. 按如下目录结构组织文档
```bash
data
├── 知识库一
│   ├── 文档一.pdf
│   └── 文档二.pdf
├── 知识库二
│   ├── 文档三.docx
│   └── 文档四.docx
└── 知识库三
    ├── 文档五.docx
    └── 文档六.pdf
```

2. 执行脚本
```bash
python import_knowledge.py --datadir=/path/to/data
```

3. 该脚本会以 data 目录下的子目录名称创建知识库，并导入该子目录下的文档到该知识库
