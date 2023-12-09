package api

import (
    "testing"
    //"net/http"
    "encoding/json"
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
    _, err := json.Marshal(data)
    if err != nil {
        t.Fail()
    }
}
