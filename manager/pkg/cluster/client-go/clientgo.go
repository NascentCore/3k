package clientgo

// NO_TEST_NEEDED

// operater cluster with client-go.
import (
	"context"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func GetObjectData(namespace, group, version, resources, name string) (map[string]interface{}, error) {
	crd := schema.GroupVersionResource{Group: group, Version: version, Resource: resources}
	data, err := dynamicClient.Resource(crd).Namespace(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return data.Object, nil
}

func GetObjects(namespace, group, version, resources string) ([]unstructured.Unstructured, error) {
	crd := schema.GroupVersionResource{Group: group, Version: version, Resource: resources}
	data, err := dynamicClient.Resource(crd).Namespace(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	return data.Items, nil
}

// !!! resources must be plural.
func ApplyWithJsonData(namespace, group, version, resources string, data map[string]interface{}) error {
	crd := schema.GroupVersionResource{Group: group, Version: version, Resource: resources}
	cro := &unstructured.Unstructured{
		Object: data,
	}
	_, err := dynamicClient.Resource(crd).Namespace(namespace).Create(context.TODO(), cro, metav1.CreateOptions{})
	return err
}

func DeleteWithName(namespace, group, version, resources, name string) error {
	crd := schema.GroupVersionResource{Group: group, Version: version, Resource: resources}
	err := dynamicClient.Resource(crd).Namespace(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{})
	return err
}

func GetNodeInfo() (*v1.NodeList, error) {
	nodes, err := k8sClient.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	return nodes, nil
}
