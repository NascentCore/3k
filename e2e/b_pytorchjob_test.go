package e2e

import (
	"context"
	"os"
	"testing"
	"time"

	tov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/e2e-framework/klient/decoder"
	"sigs.k8s.io/e2e-framework/klient/k8s"
	"sigs.k8s.io/e2e-framework/klient/k8s/resources"
	"sigs.k8s.io/e2e-framework/klient/wait"
	"sigs.k8s.io/e2e-framework/klient/wait/conditions"
	"sigs.k8s.io/e2e-framework/pkg/envconf"
	"sigs.k8s.io/e2e-framework/pkg/features"
)

func TestPytorchJob(t *testing.T) {
	featBuilder := features.New("PytorchJob")

	featBuilder.Assess("pytorch multinode", func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
		r, err := resources.New(cfg.Client().RESTConfig())
		if err != nil {
			t.Fatal(err)
		}
		if err = tov1.AddToScheme(r.GetScheme()); err != nil {
			t.Fatal(err)
		}
		r.WithNamespace(namespace)

		err = decoder.DecodeEachFile(ctx, os.DirFS("./pytorchjob"), "*", decoder.CreateHandler(r), decoder.MutateNamespace(namespace))
		if err != nil {
			t.Fatalf("Failed due to error: %s", err)
		}

		ibPytorchJob := tov1.PyTorchJob{
			ObjectMeta: metav1.ObjectMeta{Name: "pytorch-multinode-linertrain", Namespace: c.Namespace()},
		}
		err = wait.For(conditions.New(r).ResourceMatch(&ibPytorchJob, func(object k8s.Object) bool {
			pj := object.(*tov1.PyTorchJob)
			for _, cond := range pj.Status.Conditions {
				if cond.Type == tov1.JobSucceeded && cond.Status == corev1.ConditionTrue {
					return true
				}
			}
			return false
		}), wait.WithImmediate(), wait.WithInterval(10*time.Second), wait.WithTimeout(time.Duration(backOffLimitTime)*time.Minute))
		if err != nil {
			t.Fatal(err)
		}
		return ctx
	})

	featBuilder.Teardown(func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
		r, err := resources.New(cfg.Client().RESTConfig())
		if err != nil {
			t.Fail()
		}
		if err = tov1.AddToScheme(r.GetScheme()); err != nil {
			t.Fail()
		}
		r.WithNamespace(namespace)

		err = decoder.DecodeEachFile(ctx, os.DirFS("./pytorchjob"), "*", decoder.DeleteHandler(r), decoder.MutateNamespace(namespace))

		if err != nil {
			t.Fatalf("Failed teardown due to error: %s", err)
		}
		return ctx
	})

	testenv.Test(t, featBuilder.Feature())
}
