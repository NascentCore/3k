package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/yaml"
)

func main() {

	// 原始 JSON 字符串
	jsonStr := `{"max-num-seqs":4,"max-model-len":8192,"enforce-eager":""}`

	// 转义双引号,将 " 替换为 \"
	extraParams := strings.Replace(jsonStr, `"`, `\"`, -1)

	// 验证转义后的 JSON 格式是否正确
	var testMap map[string]interface{}
	if err := json.Unmarshal([]byte(strings.Replace(extraParams, `\"`, `"`, -1)), &testMap); err != nil {
		fmt.Printf("Invalid JSON format: %v\n", err)
		return
	}

	fmt.Printf("Params: %s\n", extraParams)
	// extraParamsJSON, err := json.Marshal(extraParams)
	// if err != nil {
	// 	return
	// }
	// // 转义特殊字符
	// escapedParams := strings.Replace(string(extraParamsJSON), `"`, `\"`, -1)

	serverConfigStr := fmt.Sprintf(`applications:
    - name: llm
      route_prefix: /
      import_path: vllm_app:model 
      deployments:
      - name: VLLMDeployment
        max_ongoing_requests: 5
        autoscaling_config:
          min_replicas: 1
          initial_replicas: null
          max_replicas: 1
          target_ongoing_requests: 3.0
          metrics_interval_s: 10.0
          look_back_period_s: 30.0
          smoothing_factor: 1.0
          upscale_smoothing_factor: null
          downscale_smoothing_factor: null
          upscaling_factor: null
          downscaling_factor: null
          downscale_delay_s: 600.0
          upscale_delay_s: 30.0
      runtime_env:
        working_dir: "https://sxwl-dg.oss-cn-beijing.aliyuncs.com/ray/ray_vllm/va.zip"
        env_vars:
          EXTRA_PARAMS: ''
          TENSOR_PARALLELISM: "%v"`, 4)

	fmt.Println(serverConfigStr)

	serveConfig := make(map[string]interface{})
	if err := yaml.Unmarshal([]byte(serverConfigStr), &serveConfig); err != nil {
		fmt.Printf("err: %v", err)
		return
	}

	fmt.Printf("serveConfig: %v", serveConfig)
}
