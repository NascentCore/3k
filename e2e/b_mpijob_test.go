package e2e

import (
	"context"
	"os"
	"testing"
	"time"

	mpiv2beta "github.com/kubeflow/mpi-operator/pkg/apis/kubeflow/v2beta1"
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

func TestMPIJob(t *testing.T) {
	featBuilder := features.New("MPIJobs")

	featBuilder.Assess("MPIJob mnist example", func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
		r, err := resources.New(cfg.Client().RESTConfig())
		if err != nil {
			t.Fatal(err)
		}
		if err = mpiv2beta.AddToScheme(r.GetScheme()); err != nil {
			t.Fatal(err)
		}
		r.WithNamespace(namespace)

		err = decoder.DecodeEachFile(ctx, os.DirFS("./mpijob"), "*", decoder.CreateHandler(r), decoder.MutateNamespace(namespace))
		if err != nil {
			t.Fatalf("Failed due to error: %s", err)
		}

		return ctx
	}).Assess("MPIJob Check", func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
		mpijob := mpiv2beta.MPIJob{
			ObjectMeta: metav1.ObjectMeta{Name: "tensorflow-mnist", Namespace: namespace},
		}
		t.Logf("MPIJob Checking")
		err := wait.For(conditions.New(cfg.Client().Resources()).ResourceMatch(&mpijob, func(object k8s.Object) bool {
			t.Logf("MPIJob Checking ...")
			mj := object.(*mpiv2beta.MPIJob)
			for _, cond := range mj.Status.Conditions {
				if cond.Type == mpiv2beta.JobSucceeded && cond.Status == corev1.ConditionTrue {
					t.Logf("MPIJob Check Successed")
					return true
				}
			}
			return false
		}), wait.WithImmediate(), wait.WithInterval(10*time.Second), wait.WithTimeout(time.Duration(backOffLimitTime)*time.Minute))
		if err != nil {
			t.Errorf("MPIJob Check Failed")
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

		err = decoder.DecodeEachFile(ctx, os.DirFS("./mpijob"), "*", decoder.DeleteHandler(r), decoder.MutateNamespace(namespace))

		if err != nil {
			t.Fatalf("Failed teardown due to error: %s", err)
		}
		return ctx
	})

	testenv.Test(t, featBuilder.Feature())
}
