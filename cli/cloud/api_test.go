package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
)

const URL = "http://localhost:8012/api/userJob"

func TestPostUserJob(t *testing.T) {
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
		t.Fail()
	}

	req, err := http.NewRequest("POST", URL, bytes.NewBuffer(jsonData))
	if err != nil {
		t.Fail()
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiI4NGE2NzQ3ZTU0NzI0NDk3ODk0MmIxNGYzOTZiNWNmZCIsInVzZXIiOiJqaW0xIiwic3ViIjoiamltMSJ9.RUwuHUDbL9-Xw9zsXk_dYNrche-LCsluw8FsVdd9exxOsXMwP5lRtFpYOCvn6tovjVWVkiFZ_oru_9CGmWnEsA")
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fail()
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fail()
	}
	var result map[string]interface{}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fail()
	}
	err = json.Unmarshal(body, &result)
	if err != nil {
		t.Fail()
	}
	for k, v := range result {
		fmt.Println(k, v)
	}
}
