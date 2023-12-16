package main

import (
	"encoding/json"
	"fmt"
	"sxwl/3k/pkg/communication"
	"testing"
)

func IntMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func TestIntMinBasic(t *testing.T) {
	ans := IntMin(2, -2)
	if ans != -2 {
		t.Errorf("IntMin(2, -2) = %d; want -2", ans)
	}
}
func TestJob(t *testing.T) {
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
	fmt.Printf("%+v\n", job)
}
