package utils

import (
	"errors"
	"fmt"
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
)

// NO_TEST_NEEDED
func GetModelPVC(modelname string) (string, error) {
	m, e := GetModelStorageMap()
	if e != nil {
		return "", e
	}
	if v, ok := m[modelname]; !ok {
		return "", errors.New(fmt.Sprintf("no pvc found for model %s", modelname))
	} else {
		return v, nil
	}
}

// map  modelname : pvc
func GetModelStorageMap() (map[string]string, error) {
	data, err := clientgo.GetObjects(config.CPOD_NAMESPACE, "cpod.sxwl.ai", "v1", "modelstorages")
	if err != nil {
		return nil, err
	}
	res := map[string]string{}
	for _, item := range data {
		spec, ok := item.Object["spec"].(map[string]interface{})
		if !ok {
			return nil, errors.New("no spec in data")
		}
		modelname, ok := spec["modelname"].(string)
		if !ok {
			return nil, errors.New("no modelname in spec")
		}
		pvc, ok := spec["pvc"].(string)
		if !ok {
			return nil, errors.New("no pvc in spec")
		}
		res[modelname] = pvc
	}
	return res, nil
}
