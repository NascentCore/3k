package controller

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestCheckPVCExists(t *testing.T) {
	// 初始化测试用的客户端
	ctx := context.Background()
	pvcName := "logs"
	namespace := "default"

	// 创建一个 Scheme，这里假设你已经注册了所有需要的类型
	sch := scheme.Scheme
	sch.AddKnownTypes(corev1.SchemeGroupVersion, &corev1.PersistentVolumeClaim{})

	// 创建一个存在的 PVC 对象
	existingPVC := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pvcName,
			Namespace: namespace,
		},
	}

	// 使用 fake client 并注入存在的 PVC
	client := fake.NewClientBuilder().WithScheme(sch).WithObjects(existingPVC).Build()

	// 创建你的 Reconciler 实例，注入 fake client
	reconciler := &CPodJobReconciler{
		Client: client,
	}

	// 测试存在的情况
	exists, err := reconciler.checkPVCExists(ctx, pvcName, namespace)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !exists {
		t.Errorf("expected PVC to exist")
	}

	// 测试不存在的情况
	notExists, err := reconciler.checkPVCExists(ctx, "non-existent-pvc", namespace)
	if err != nil {
		t.Errorf("unexpected error for non-existent PVC: %v", err)
	}
	if notExists {
		t.Errorf("expected PVC not to exist")
	}
}
