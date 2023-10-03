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

// init dynamicClient.
func init() {
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

// try to create the out-cluster config.
func outClusterConfig() (*rest.Config, error) {
	filename := ""
	if kubeconfigFile == "" {
		// not provided , use conventional path
		if home := homedir.HomeDir(); home != "" {
			filename = filepath.Join(home, ".kube", "config")
		} else {
			return nil, errors.New("no kubeconfig file")
		}
	} else {
		filename = kubeconfigFile
	}
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return clientcmd.RESTConfigFromKubeConfig(b)
}
