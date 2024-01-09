# llama-2-7b 微调训练及推理部署

1. 下载获取llama-2-7b base model

```bash
# kubectl -n cpod get modelstorage
NAME        MODELTYPE     MODELNAME   PHASE
llama2-7b   huggingface   llama2-7b   done

```


2. 提交微调训练任务cpodjob

```bash
# kubectl -n cpod create -f cpodjob.yaml
```

等待训练任务完成

```bash
# kubectl -n cpod get pytorchjob
NAME         STATE       AGE
llama-2-7b   Succeeded   42m

# kubectl -n cpod get cpodjob  llama-2-7b -oyaml | less
```

3. 基于微调训练模型部署推理API服务

```bash

# kubectl create -f inferenceservice.yaml

```

4. 验证

```bash
# curl http://10.233.24.14/generate     -d '{
        "prompt": "Beijing is a",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
    }'
{"text":["Beijing is a language model developed by Large Model Systems Organization (LMSYS).","Beijing is a language model trained by researchers from Large Model Systems Organization (LMS","Beijing is a language model created by researchers from Large Model Systems Organization (LMS","Beijing is a language model meticulously developed by Large Model Systems Organization (LMS"]}
```