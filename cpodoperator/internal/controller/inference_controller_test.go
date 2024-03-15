package controller

import (
	"context"
	"testing"

	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	netv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestInferenceReconciler_DeployWebUI tests the DeployWebUI method
func TestInferenceReconciler_DeployWebUI(t *testing.T) {
	// Initialize the scheme
	s := scheme.Scheme
	if err := cpodv1beta1.AddToScheme(s); err != nil {
		t.Fatalf("Unable to add cpodv1beta1 scheme: (%v)", err)
	}
	if err := appsv1.AddToScheme(s); err != nil {
		t.Fatalf("Unable to add appsv1 scheme: (%v)", err)
	}

	// Create a new fake client
	cl := fake.NewClientBuilder().WithScheme(s).Build()

	// Create an instance of InferenceReconciler
	reconciler := InferenceReconciler{
		Client: cl,
		Scheme: s,
	}

	// Define the Inference object
	inference := &cpodv1beta1.Inference{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-inference",
			Namespace: "default",
		},
	}

	// Call DeployWebUI
	err := reconciler.DeployWebUI(context.TODO(), inference)
	if err != nil {
		t.Errorf("DeployWebUI returned an error: %v", err)
	}

	// Verify that the Deployment and Service were created
	deployment := &appsv1.Deployment{}
	err = cl.Get(context.TODO(), client.ObjectKey{
		Name:      "test-inference-web-ui",
		Namespace: "default",
	}, deployment)
	if err != nil {
		t.Errorf("Failed to get Deployment: %v", err)
	}

	service := &corev1.Service{}
	err = cl.Get(context.TODO(), client.ObjectKey{
		Name:      "test-inference-web-ui",
		Namespace: "default",
	}, service)
	if err != nil {
		t.Errorf("Failed to get Service: %v", err)
	}
}

// TestInferenceReconciler_DeployWebUIIngress tests the DeployWebUIIngress method
func TestInferenceReconciler_DeployWebUIIngress(t *testing.T) {
	// Initialize the scheme
	s := scheme.Scheme
	if err := cpodv1beta1.AddToScheme(s); err != nil {
		t.Fatalf("Unable to add cpodv1beta1 scheme: (%v)", err)
	}
	if err := netv1.AddToScheme(s); err != nil {
		t.Fatalf("Unable to add netv1 scheme: (%v)", err)
	}

	// Create a new fake client
	cl := fake.NewClientBuilder().WithScheme(s).Build()

	// Create an instance of InferenceReconciler
	reconciler := InferenceReconciler{
		Client: cl,
		Scheme: s,
	}

	// Define the Inference object
	inference := &cpodv1beta1.Inference{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-inference",
			Namespace: "default",
		},
	}

	// Call DeployWebUIIngress
	err := reconciler.DeployWebUIIngress(context.TODO(), inference)
	if err != nil {
		t.Errorf("DeployWebUIIngress returned an error: %v", err)
	}

	// Verify that the Ingress was created
	ingress := &netv1.Ingress{}
	err = cl.Get(context.TODO(), client.ObjectKey{
		Name:      "test-inference-web-ui-ingress",
		Namespace: "default",
	}, ingress)
	if err != nil {
		t.Errorf("Failed to get Ingress: %v", err)
	}
}
