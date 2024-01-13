package config

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

// 测试 GetConfigMapValues 函数
func TestGetConfigMapValues(t *testing.T) {
	// 创建一个假的 ConfigMap
	fakeConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cpod-info",
			Namespace: "default",
		},
		Data: map[string]string{
			"a": "valueA",
			"b": "valueB",
			"c": "valueC",
			"d": "valueD",
		},
	}

	// 创建一个假的 Kubernetes 客户端，并将假的 ConfigMap 添加到客户端中
	fakeClient := fake.NewSimpleClientset(fakeConfigMap)

	// 调用函数
	result, err := GetConfigMapValues(fakeClient, "cpod-info", "default")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// 验证返回的结果是否正确
	expected := map[string]string{"a": "valueA", "b": "valueB", "c": "valueC", "d": "valueD"}
	if !equalMaps(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

// 辅助函数，用于比较两个 map 是否相等
func equalMaps(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}

	for k, v := range a {
		if v2, ok := b[k]; !ok || v != v2 {
			return false
		}
	}

	return true
}
