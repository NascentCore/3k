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

func TestGPT3(t *testing.T) {
	feature := features.New("GPT3 check").
		Assess("check 1h1g", func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
			r, err := resources.New(c.Client().RESTConfig())
			if err != nil {
				t.Fatal(err)
			}
			if err = tov1.AddToScheme(r.GetScheme()); err != nil {
				t.Fatal(err)
			}
			r.WithNamespace(namespace)

			err = decoder.DecodeEachFile(ctx, os.DirFS("./model/gpt3-1.3b"), "*", decoder.CreateHandler(r), decoder.MutateNamespace(namespace))

			if err != nil {
				t.Fatalf("Failed due to error: %s", err)
			}

			gpt3PytorchJob := tov1.PyTorchJob{
				ObjectMeta: metav1.ObjectMeta{Name: "gpt3-1h1g", Namespace: c.Namespace()},
			}

			// 通过日志判断是否成功，目前等待训练完成需要等待很长时间
			err = wait.For(conditions.New(r).ResourceMatch(&gpt3PytorchJob, func(object k8s.Object) bool {
				pj := object.(*tov1.PyTorchJob)
				for _, cond := range pj.Status.Conditions {
					if cond.Type == tov1.JobRunning && cond.Status == corev1.ConditionTrue {
						return true
					}
				}
				return false
			}), wait.WithInterval(10*time.Second), wait.WithTimeout(time.Duration(backOffLimitTime)*time.Minute))
			if err != nil {
				t.Fatal(err)
			}
			return ctx
		}).
		Teardown(func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
			r, err := resources.New(cfg.Client().RESTConfig())
			if err != nil {
				t.Fail()
			}
			if err = tov1.AddToScheme(r.GetScheme()); err != nil {
				t.Fail()
			}
			r.WithNamespace(namespace)

			err = decoder.DecodeEachFile(ctx, os.DirFS("./model/gpt3-1.3b"), "*", decoder.DeleteHandler(r), decoder.MutateNamespace(namespace))

			if err != nil {
				t.Fatalf("Failed due to error: %s", err)
			}
			return ctx
		}).Feature()

	testenv.Test(t, feature)

}
