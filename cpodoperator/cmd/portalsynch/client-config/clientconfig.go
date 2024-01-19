package clientconfig

// NO_TEST_NEEDED

import (
	"errors"
	"os"
	"path/filepath"

	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

// try to create the out-cluster config.
func outClusterConfig() (*rest.Config, error) {
	filename := ""
	if home := homedir.HomeDir(); home != "" {
		filename = filepath.Join(home, ".kube", "config")
	} else {
		return nil, errors.New("no kubeconfig file")
	}

	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return clientcmd.RESTConfigFromKubeConfig(b)
}

func GetClientConfig() *rest.Config {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		cfg, err = outClusterConfig()
		if err != nil {
			panic(err)
		}
	}
	return cfg
}
