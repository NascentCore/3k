package utils

import (
	"errors"
	"fmt"
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
)

// NO_TEST_NEEDED
func GetDatasetPVC(datasetname string) (string, error) {
	m, e := GetDatasetStorageMap()
	if e != nil {
		return "", e
	}
	if v, ok := m[datasetname]; !ok {
		return "", errors.New(fmt.Sprintf("no pvc found for dataset %s", datasetname))
	} else {
		return v, nil
	}
}

// map  datasetname : pvc
func GetDatasetStorageMap() (map[string]string, error) {
	data, err := clientgo.GetObjects(config.CPOD_NAMESPACE, "cpod.sxwl.ai", "v1", "datasetstorages")
	if err != nil {
		return nil, err
	}
	res := map[string]string{}
	for _, item := range data {
		spec, ok := item.Object["spec"].(map[string]interface{})
		if !ok {
			return nil, errors.New("no spec in data")
		}
		datasetname, ok := spec["datasetname"].(string)
		if !ok {
			return nil, errors.New("no datasetname in spec")
		}
		pvc, ok := spec["pvc"].(string)
		if !ok {
			return nil, errors.New("no pvc in spec")
		}
		res[datasetname] = pvc
	}
	return res, nil
}
