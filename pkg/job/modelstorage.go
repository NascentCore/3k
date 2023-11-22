package job

import (
	"errors"
	"fmt"
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
)

// NO_TEST_NEEDED
func GetModelPVC(modelurl string) (string, error) {
	m, e := GetModelStorageMap()
	if e != nil {
		return "", e
	}
	if v, ok := m[modelurl]; !ok {
		return "", errors.New(fmt.Sprintf("no pvc found for model %s", modelurl))
	} else {
		return v, nil
	}
}

// map  modelurl : pvc
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
		modelurl, ok := spec["modelurl"].(string)
		if !ok {
			return nil, errors.New("no modelurl in spec")
		}
		pvc, ok := spec["pvc"].(string)
		if !ok {
			return nil, errors.New("no pvc in spec")
		}
		res[modelurl] = pvc
	}
	return res, nil
}
