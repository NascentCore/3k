package cpod_cache

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	divider = "%%%%"
)

// Encode 把dataType和dataId合并到一个字符串内
func Encode(dataType int64, dataId string) string {
	return fmt.Sprintf("%d%s%s", dataType, divider, dataId)
}

// Decode 把encoded字符串解析成 dataType和dataId
func Decode(encoded string) (int64, string, error) {
	parts := strings.Split(encoded, divider)
	if len(parts) != 2 {
		return 0, "", fmt.Errorf("invalid encoded string")
	}

	dataType, err := strconv.ParseInt(parts[0], 10, 64)
	if err != nil {
		return 0, "", fmt.Errorf("invalid data type: %v", err)
	}

	dataId := parts[1]
	return dataType, dataId, nil
}
