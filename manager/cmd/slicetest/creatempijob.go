package main

import clientgo "sxwl/3k/manager/pkg/cluster/client-go"

// cross compile
// CGO_ENABLED=0  GOOS=linux  GOARCH=amd64  go build main.go
// ssh -p 60022 peiqing@219.159.22.20
func main() {
	data := map[string]interface{}{
		"apiVersion": "sxwl.ai/v1",
		"kind":       "MngJob",
		"metadata": map[string]interface{}{
			"name": "secondjob", //!!!must be subdomain
		},
		"spec": map[string]interface{}{
			"replicas": 2,
			"jobName":  "my second job",
			"image":    "my awesome image",
		},
	}
	err := clientgo.ApplyWithJsonData("cpq", "sxwl.ai", "v1", "mngjobs", data)
	if err != nil {
		panic(err)
	}
}
