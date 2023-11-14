package clientgo

// NO_TEST_NEEDED

// operater cluster with client-go.
import (
	"context"
	"errors"
	"fmt"
	"sxwl/3k/manager/pkg/log"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	apiv1 "k8s.io/api/batch/v1"
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

func GetPVC(name, namespace string) (*v1.PersistentVolumeClaim, error) {
	return k8sClient.CoreV1().PersistentVolumeClaims(namespace).Get(context.TODO(), name, metav1.GetOptions{})
}

func CreatePVCIFNotExist(name, namespace, sc, accessMode string, MBs int) error {
	// TODO:  use k8sClient
	if accessMode != "ReadWriteMany" && accessMode != "ReadWriteOnce" && accessMode != "ReadWriteOncePod" {
		return errors.New("invalid accessMode")
	}
	// check existence
	_, err := GetPVC(name, namespace)
	if err == nil { //如果可以取得，代表PVC一定存在
		log.SLogger.Infow("pvc is already there", "pvc name", name)
		return nil
	}

	// TODO: adjust
	if MBs > 1024*100 {
		return errors.New("request volume too large")
	}
	if MBs <= 0 {
		return errors.New("request volume too small")
	}
	return ApplyWithJsonData(namespace, "", "v1", "persistentvolumeclaims",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "PersistentVolumeClaim",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"accessModes": []interface{}{
					accessMode,
				},
				"resources": map[string]interface{}{
					"requests": map[string]interface{}{
						"storage": fmt.Sprintf("%dMi", MBs),
					},
				},
				"storageClassName": sc,
				"volumeMode":       "Filesystem",
			},
		})
}

func GetK8SJobs(namespace string) (*apiv1.JobList, error) {
	jobs, err := k8sClient.BatchV1().Jobs(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	return jobs, err
}

func DeleteK8SJob(namespace, name string) error {
	t := metav1.DeletePropagationBackground
	return k8sClient.BatchV1().Jobs(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{
		PropagationPolicy: &t,
	})
}

func DeletePVC(namespace, name string) error {
	return k8sClient.CoreV1().PersistentVolumeClaims(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{})
}
