package fs

import (
	"encoding/json"
	"fmt"
	"os"
)

func PreviewJSONArray(filePath string, limit int) (string, error) {
	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// 创建decoder
	decoder := json.NewDecoder(file)

	// 读取开始的数组标记 '['
	_, err = decoder.Token()
	if err != nil {
		return "", fmt.Errorf("expected array start token: %v", err)
	}

	// 用于存储前N个元素
	var items []json.RawMessage
	count := 0

	// 读取指定数量的元素
	for decoder.More() && count < limit {
		var item json.RawMessage
		if err := decoder.Decode(&item); err != nil {
			return "", fmt.Errorf("failed to decode item: %v", err)
		}
		items = append(items, item)
		count++
	}

	// 将收集到的元素编码为JSON数组
	result, err := json.Marshal(items)
	if err != nil {
		return "", fmt.Errorf("failed to marshal preview items: %v", err)
	}

	return string(result), nil
}

func CountJSONArray(filePath string) (int64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return 0, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// 使用 json.Decoder 来确保正确解析 JSON
	decoder := json.NewDecoder(file)

	// 读取开始的数组标记 '['
	t, err := decoder.Token()
	if err != nil {
		return 0, fmt.Errorf("expected array start token: %v", err)
	}
	if delim, ok := t.(json.Delim); !ok || delim != '[' {
		return 0, fmt.Errorf("expected array start '['")
	}

	var count int64
	// 只要还有更多的元素就继续读取
	for decoder.More() {
		// 跳过当前值，只是为了移动到下一个元素
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return 0, fmt.Errorf("failed to decode element: %v", err)
		}
		count++
	}

	// 读取结束的数组标记 ']'
	if _, err := decoder.Token(); err != nil {
		return 0, fmt.Errorf("expected array end token: %v", err)
	}

	return count, nil
}
