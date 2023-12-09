package api

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "testing"
)

const URL = "https://llm.nascentcore.cn/api/userJob"
func TestPostUserJob(t *testing.T) {
    data := map[string]interface{}{
            "gpuNumber": 1,
            "gpuType": "NVIDIA-GeForce-RTX-3090",
            "ckptPath": "/data",
            "ckptVol": "10000",
            "modelPath": "/data2",
            "modelVol": "10000",
            "imagePath": "dockerhub.kubekey.local/kubesphereio/modihand:test",
            "jobType": "GeneralJob",
            "stopType": "1",
            "stopTime": 5,
            "pretrainedModelName": "chatglm3-6b",
            "pretrainedModelPath": "/sixpen_models/chatlm3",
            "datasetName": "modihand-dataset",
            "datasetPath": "/sixpen_models/modihand_outputs/test_10059997",
            "runCommand": "sleep 600",
            "callbackUrl": "https://aiapi.yangapi.cn/api/test/callback",
            "env": "\"MODIHAND_OPEN_NODE_TOKEN\": \"9999\"",
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
    //不能checkin 内部测试token不能暴露到互联网上，容易泄露内部测试环境
    req.Header.Set("Authorization", "Bearer eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJiNjdhMjAwNzljZDY0NmE4YThlNWM3ZDY1ZDJhOTcxNSIsInVzZXIiOiJqaW0xIiwic3ViIjoiamltMSJ9.K_8kcdXOUhoDIp5pZX48MC1g6Q_Ut0zXVO5V2heh3AgJpO9BJqjix61lM3Z_gEK4L133MFkQbtE9EiY0HyfM1A")
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        t.Fail()
    }
    defer resp.Body.Close()
    var result map[string]interface{}
    body,err := io.ReadAll(resp.Body)
    if err != nil {
        t.Fail()
    }
    print("body",body)
    err = json.Unmarshal(body, &result)
    if err != nil {
        t.Fail()
    }
    for k,v := range result{
        fmt.Println(k,v)
    }
    //println("result",result)
}
