package e2e

import (
	"context"
	"os"
	"testing"
	"time"

	tov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/e2e-framework/klient/decoder"
	"sigs.k8s.io/e2e-framework/klient/k8s"
	"sigs.k8s.io/e2e-framework/klient/k8s/resources"
	"sigs.k8s.io/e2e-framework/klient/wait"
	"sigs.k8s.io/e2e-framework/klient/wait/conditions"
	"sigs.k8s.io/e2e-framework/pkg/envconf"
	"sigs.k8s.io/e2e-framework/pkg/features"
)

func TestIBNode(t *testing.T) {
	r, err := resources.New(cfg.Client().RESTConfig())
	if err != nil {
		t.Fatal(err)
		return
	}
	if err = tov1.AddToScheme(r.GetScheme()); err != nil {
		t.Fatal(err)
		return
	}
	r.WithNamespace(namespace)
	// Get IB Node count
	var ibNodes corev1.NodeList
	err = r.List(context.Background(), &ibNodes, resources.WithLabelSelector(labels.FormatLabels(map[string]string{"feature.node.kubernetes.io/rdma.capable": "true"})))
	if err != nil {
		t.Fatal(err)
		return
	}
	ibNodeCount := int32(len(ibNodes.Items))
	if ibNodeCount == 0 || ibNodeCount == 1 {
		t.Logf("No IB node found")
		// 不再执行该feature
		return
	}

	feature := features.New("IB Check").
		Setup(func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
			err = decoder.DecodeEachFile(ctx, os.DirFS("./ib"), "*", decoder.CreateHandler(r), decoder.MutateNamespace(namespace), decoder.MutateOption(func(obj k8s.Object) error {
				pj := obj.(*tov1.PyTorchJob)
				pj.Spec.PyTorchReplicaSpecs[tov1.PyTorchJobReplicaTypeWorker].Replicas = &ibNodeCount
				return nil
			}))
			if err != nil {
				t.Fatalf("Failed due to error: %s", err)
			}

			return ctx
		}).
		Assess("Check ib communication", func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
			client, err := c.NewClient()
			if err != nil {
				t.Fatal(err)
			}
			ibPytorchJob := tov1.PyTorchJob{
				ObjectMeta: metav1.ObjectMeta{Name: "ib-check", Namespace: c.Namespace()},
			}
			err = wait.For(conditions.New(client.Resources()).ResourceMatch(&ibPytorchJob, func(object k8s.Object) bool {
				pj := object.(*tov1.PyTorchJob)
				for _, cond := range pj.Status.Conditions {
					if cond.Type == tov1.JobSucceeded && cond.Status == corev1.ConditionTrue {
						t.Logf("IB Check Successed")
						return true
					}
				}
				return false
			}), wait.WithImmediate(), wait.WithInterval(10*time.Second), wait.WithTimeout(2*time.Minute))
			if err != nil {
				t.Errorf("IB Check failed")
				// t.Fatal(err)
			}
			return ctx
		}).
		Teardown(func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
			err = decoder.DecodeEachFile(ctx, os.DirFS("./ib"), "*", decoder.DeleteHandler(r), decoder.MutateNamespace(namespace))
			if err != nil {
				t.Fatalf("Failed due to error: %s", err)
			}
			return ctx
		}).
		Feature()

	if enableIb {
		testenv.Test(t, feature)
	}
}
