package e2e

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/e2e-framework/pkg/envconf"
	"sigs.k8s.io/e2e-framework/pkg/features"
)

func TestListPods(t *testing.T) {
	f := features.New("pod list").
		Assess("pods from kube-system", func(ctx context.Context, t *testing.T, cfg *envconf.Config) context.Context {
			client, err := cfg.NewClient()
			if err != nil {
				t.Fatal(err)
			}
			var pods corev1.PodList
			err = client.Resources("kube-system").List(context.TODO(), &pods)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("found %d pods", len(pods.Items))
			if len(pods.Items) == 0 {
				t.Fatal("no pods in namespace kube-system")
			}
			return ctx
		})

	testenv.Test(t, f.Feature())
}
