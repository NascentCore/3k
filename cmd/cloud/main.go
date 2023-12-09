package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sxwl/3k/pkg/log"
)

func main() {
	// TODO: 使用 Cobra 来编写完整的命令行功能 以后应该使用专门的命令行工具框架库
	// https://github.com/spf13/cobra
	if len(os.Args) != 3 {
		fmt.Println("Usage : sxwlapitest  [ need createjob url ] [token]")
		os.Exit(1)
	}
	url := os.Args[1]
	token := os.Args[2]
	// TODO: 添加 --json 选项，让用户输入 Json 字符串
	data := map[string]interface{}{
		"gpuNumber":           1,
		"gpuType":             "NVIDIA-GeForce-RTX-3090",
		"ckptPath":            "/data",
		"ckptVol":             "10000",
		"modelPath":           "/data2",
		"modelVol":            "10000",
		"imagePath":           "dockerhub.kubekey.local/kubesphereio/modihand:test",
		"jobType":             "GeneralJob",
		"stopType":            "1",
		"stopTime":            5,
		"pretrainedModelName": "chatglm3-6b",
		"pretrainedModelPath": "/sixpen_models/chatlm3",
		"datasetName":         "modihand-dataset",
		"datasetPath":         "/sixpen_models/modihand_outputs/test_10059997",
		"runCommand":          "sleep 600",
		"callbackUrl":         "https://aiapi.yangapi.cn/api/test/callback",
		"env":                 "\"MODIHAND_OPEN_NODE_TOKEN\": \"9999\"",
	}
	jsonData, err := json.Marshal(data)
	if err != nil {
		log.Logger.Error(err.Error())
		os.Exit(1)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Logger.Error(err.Error())
		os.Exit(1)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Logger.Error(err.Error())
		os.Exit(1)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		log.Logger.Error(err.Error())
		os.Exit(1)
	}
	var result map[string]interface{}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Logger.Error(err.Error())
		os.Exit(1)
	}
	err = json.Unmarshal(body, &result)
	if err != nil {
		log.Logger.Error(err.Error())
		os.Exit(1)
	}
	for k, v := range result {
		fmt.Println(k, v)
	}
}
