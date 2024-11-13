package sxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/spf13/viper"
)

type Resource struct {
	ResourceID   string `json:"resource_id"`
	ResourceType string `json:"resource_type"`
	ResourceName string `json:"resource_name"`
	ResourceSize int64  `json:"resource_size"`
	IsPublic     bool   `json:"is_public"`
	UserID       string `json:"user_id"`
	Meta         string `json:"meta"`
}

func AddResource(token string, resource Resource) error {
	// Convert resource to JSON
	resourceJSON, err := json.Marshal(resource)
	if err != nil {
		return fmt.Errorf("error marshaling resource to JSON: %s", err)
	}

	// Create a new request using http.NewRequest
	req, err := http.NewRequest(http.MethodPost, viper.GetString("resource_add_url"), bytes.NewBuffer(resourceJSON))
	if err != nil {
		return fmt.Errorf("error creating request: %s", err)
	}

	// Add an 'Authorization' header to the request
	req.Header.Add("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	// Send the request using http.Client
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("error sending request to API endpoint: %s", err)
	}
	defer resp.Body.Close()

	// Check if the response status code indicates success (200 OK)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("request failed with status code: %d", resp.StatusCode)
	}

	return nil
}

// LoadResourceReq 加载资源请求
type LoadResourceReq struct {
	Source       string `json:"source"`
	ResourceID   string `json:"resource_id"`
	ResourceType string `json:"resource_type"`
	Meta         string `json:"meta"`
}

// LoadResource 调用resource/load接口
func LoadResource(token string, req LoadResourceReq) error {
	// 转换请求为JSON
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("error marshaling request to JSON: %s", err)
	}

	// 创建HTTP请求
	httpReq, err := http.NewRequest(http.MethodPost, viper.GetString("resource_load_url"), bytes.NewBuffer(reqJSON))
	if err != nil {
		return fmt.Errorf("error creating request: %s", err)
	}

	// 添加请求头
	httpReq.Header.Add("Authorization", "Bearer "+token)
	httpReq.Header.Set("Content-Type", "application/json")

	// 发送请求
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("error sending request to API endpoint: %s", err)
	}
	defer resp.Body.Close()

	// 读取响应body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("error reading response body: %s", err)
	}

	// 检查响应状态码
	if resp.StatusCode != http.StatusOK {
		// 尝试解析错误响应
		var errResp struct {
			Message string `json:"message"`
		}
		if err := json.Unmarshal(body, &errResp); err == nil {
			return fmt.Errorf("request failed: message=%s", errResp.Message)
		}
		// 如果无法解析JSON,则返回原始响应
		return fmt.Errorf("request failed with status code %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

type ResourceTaskStatus struct {
	ResourceID   string `json:"resource_id"`
	ResourceType string `json:"resource_type"`
	Source       string `json:"source"`
	Status       string `json:"status"`
}

type ResourceTaskStatusResp struct {
	Data  []ResourceTaskStatus `json:"data"`
	Total int64                `json:"total"`
}

// GetResourceSyncTaskStatus 查询资源同步任务状态
func GetResourceSyncTaskStatus(token string) (*ResourceTaskStatusResp, error) {
	// 发送GET请求
	req, err := http.NewRequest(http.MethodGet, viper.GetString("resource_task_status_url"), nil)
	if err != nil {
		return nil, fmt.Errorf("create request failed: %v", err)
	}

	// 设置请求头
	req.Header.Set("Sx-User-ID", token)

	// 发送请求
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response failed: %v", err)
	}

	// 解析响应
	var result ResourceTaskStatusResp
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("parse response failed: %v", err)
	}

	return &result, nil
}
