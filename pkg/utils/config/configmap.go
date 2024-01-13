package config

// Reading configMap as config entries

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
)

type K8sClient interface {
	CoreV1() corev1.CoreV1Interface
}

// GetConfigMapValues 读取指定 ConfigMap 中的数据并返回一个 map[string]string
func GetConfigMapValues(client K8sClient, configMapName, namespace string) (map[string]string, error) {
	// 从 Kubernetes API 读取 ConfigMap
	cm, err := client.CoreV1().ConfigMaps(namespace).Get(context.Background(), configMapName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	return cm.Data, nil
}
