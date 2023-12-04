package e2e

import (
	"context"
	"testing"

	"sigs.k8s.io/e2e-framework/pkg/envconf"
	"sigs.k8s.io/e2e-framework/pkg/features"
)

func TestGPT3(t *testing.T) {
	feature := features.New("GPT3 check").
		Setup(func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
			// create workload
			return ctx
		}).
		Assess("check 1h1g", func(ctx context.Context, t *testing.T, c *envconf.Config) context.Context {
			return ctx
		}).
		Feature()

	testenv.Test(t, feature)

}
