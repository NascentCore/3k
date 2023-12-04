package e2e

import (
	"context"
	"flag"
	"log"
	"os"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/e2e-framework/klient/k8s/resources"
	"sigs.k8s.io/e2e-framework/pkg/env"
	"sigs.k8s.io/e2e-framework/pkg/envconf"
	"sigs.k8s.io/e2e-framework/pkg/envfuncs"
)

var (
	testenv   env.Environment
	cfg       envconf.Config
	namespace string
	enableIb  bool
)

func TestMain(m *testing.M) {
	// declare the test environment
	flag.BoolVar(&enableIb, "enable-ib", true, "whethe check ib")
	flag.StringVar(&namespace, "namespace", "3k-e2e", "whethe check ib")

	cfg, err := envconf.NewFromFlags()
	if err != nil {
		log.Fatalf("failed to build envconf from flags: %v", err)
	}
	cfg.WithKubeconfigFile("/Users/donggang/.kube/configs/sxwl-master")
	testenv = env.NewWithConfig(cfg)

	testenv.Setup(
		envfuncs.CreateNamespace(namespace),
		// 获取集群GPU数量
	)

	testenv.Finish(
		envfuncs.DeleteNamespace(namespace),
	)

	os.Exit(testenv.Run(m))
}

type Node struct {
	Name       string
	GPUCount   int
	GPUProduct string
}

// CountGPU 一些case对资源数量有一定要求，需要感知集群GPU topo
func CountGPU(ctx context.Context) ([]Node, error) {
	r, err := resources.New(cfg.Client().RESTConfig())
	if err != nil {
		return nil, err
	}
	var nodes corev1.NodeList
	err = r.List(ctx, &nodes)
	if err != nil {
		return nil, err
	}
	var res []Node
	for _, node := range nodes.Items {
		for resourceName, cap := range node.Status.Capacity {
			if resourceName == "nvidia.com/gpu" {
				res = append(res, Node{
					Name:       node.Name,
					GPUCount:   cap.Size(),
					GPUProduct: node.Labels["nvidia.com/gpu.product"],
				})
			}
		}
	}
	return res, nil
}
