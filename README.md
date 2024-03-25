# 3K

[![codecov](https://codecov.io/gh/NascentCore/3k/graph/badge.svg?token=7L2HQJ3BSP)](https://codecov.io/gh/NascentCore/3k)

## 简介

三千平台是云原生大模型训推平台，得名于其 3 个核心指标：
* 千卡：支持千卡 A100 或等效智算集群
* 千亿参数：支持千亿参数大模型训练、推理
* 千小时：支持千小时以上无人干预大模型训练

Named after 3 major performance metrics of the system:
* 1000+ GPUs
* 100B+ Transformer model
* 1000+ hours uninterrupted training

## 名词解释

| Acronym  |      Meaning  |  涵义 |
|----------|:-------------:|:------|
| 1g   | 1 GPU           | 1 卡 |
| 1h1g | 1 node 1 GPU    | 1 机 1 卡 |
| 1h8g | 1 node 8 GPU    | 1机8卡  |
| 2h8g | 2 nodes 16 GPUs | 2机16卡 |

## SuperLinter

```shell
tools/super_linter.sh <file-or-directory-to-be-linted>
```

## 开源奖励计划

算想未来每年提供一定数量的资金，支持首次进入开源社区的学生、工程师。

### 2024 年

20,000 元人民币用于支持开源社区参与算想未来三千平台的各项研发工作；包括：
* [三千平台](https://github.com/NascentCore/3k)
* [CPodOperator](https://github.com/NascentCore/cpodoperator)
* [blog.sxwl.ai](https://github.com/NascentCore/blog.sxwl.ai)
* [docs.sxwl.ai](https://github.com/NascentCore/docs.sxwl.ai)

所有参与以上项目工作的社区成员都可以获得每小时 100 元人民币的资助，根据参与以上项目的实际耗时。
请在开始工作之前与 [@yzhao-2023](https://github.com/yzhao-2023) 联系以获得指导。

## 安装 SLO

* 1 小时以内完成安装，即从开始安装 1 小时以内三千平台（不包含大模型相关数据资产）完成安装；评判标准：可以开始运行 IB 测试任务、bert 任务
* 3 小时内完成 LLaMA2-7B 数据资产安装；即从开始安装 3 小时以内 LLaMA2-7B 模型、容器镜像、数据集完成安装；评判标准：可以开始运行 LLaMA2-7B 预训练、微调、推理演示
* 24 小时内完成 10 个主流的开源大模型的数据资产；即从开始安装 24 小时、评判标准：可以开始运行任意模型的微调、推理演示
