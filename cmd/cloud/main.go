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
	// TODO: with cli tools
	if len(os.Args) != 3 {
		fmt.Println("Usage : sxwlapitest  [ need createjob url ] [token]")
		os.Exit(1)
	}
	url := os.Args[1]
	token := os.Args[2]
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
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Logger.Error(err.Error())
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Logger.Error(err.Error())
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		log.Logger.Error(err.Error())
	}
	var result map[string]interface{}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Logger.Error(err.Error())
	}
	err = json.Unmarshal(body, &result)
	if err != nil {
		log.Logger.Error(err.Error())
	}
	for k, v := range result {
		fmt.Println(k, v)
	}
}
