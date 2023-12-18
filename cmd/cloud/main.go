package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"sxwl/3k/pkg/communication"
	"sxwl/3k/pkg/log"
)

func main() {
	// TODO: 使用 Cobra 来编写完整的命令行功能 以后应该使用专门的命令行工具框架库
	// https://github.com/spf13/cobra
	if len(os.Args) != 5 {
		fmt.Println("Usage : sxwlapitest  [ need createjob url ] [gpunumber] [gputype] [token]")
		os.Exit(1)
	}

	// TODO: 使用命令行参数 https://gobyexample.com/command-line-flags
	url := os.Args[1]
	token := os.Args[4]
	gpuNum := os.Args[2]
	gpuType := os.Args[3]
	jobStr := `
        {
	        "jobName": "ai1b4f3a3f-17f2-4257-a1e8-f49e1641176a",
	        "pretrainedModelName": "chatglm3-6b",
	        "imagePath": "dockerhub.kubekey.local/kubesphereio/sxwl-ai/bert:withcmd",
	        "ckptPath": "/data",
	        "datasetName": "",
	        "modelPath": "/model",
	        "stopType": 0,
	        "env": {
	            "TESTENV": "1111"
	        },
	        "modelVol": 100,
	        "datasetPath": "",
	        "gpuNumber": 1,
	        "pretrainedModelPath": "/pretrainedModel",
	        "ckptVol": 100,
	        "runCommand": "sleep 300",
	        "callbackUrl": "https://cloud.nascentcore.cn/api/test/callback",
	        "jobType": "GeneralJob",
	        "gpuType": "NVIDIA-GeForce-RTX-3090"
        }`
	job := communication.RawJobDataItem{}
	json.Unmarshal([]byte(jobStr), &job)
	gpuCnt, _ := strconv.Atoi(gpuNum)
	job.GpuNumber = gpuCnt
	job.GpuType = gpuType
	jsonD, err := json.Marshal(job)
	if err != nil {
		log.Logger.Error(err.Error())
		os.Exit(1)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonD))
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
