package clientgo

// NO_TEST_NEEDED

import (
	"errors"
	"os"
	"path/filepath"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

var (
	dynamicClient  *dynamic.DynamicClient
	k8sClient      *kubernetes.Clientset
	kubeconfigFile string
)

// init k8s client.
// this func must be called before u can do any thing related to k8s.
// prefer called in main
func InitClient() {
	var cfg *rest.Config
	var err error
	cfg, err = inClusterConfig()
	if err != nil {
		cfg, err = outClusterConfig()
		if err != nil {
			panic(err)
		}
	}
	dynamicClient, err = dynamic.NewForConfig(cfg)
	if err != nil {
		panic(err)
	}
	k8sClient, err = kubernetes.NewForConfig(cfg)
	if err != nil {
		panic(err)
	}
}

func SetKubeConfigFile(filename string) {
	kubeconfigFile = filename
}

// try to create the in-cluster config.
func inClusterConfig() (*rest.Config, error) {
	return rest.InClusterConfig()
}
