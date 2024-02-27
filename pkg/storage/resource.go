package storage

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"strings"
	"sxwl/3k/pkg/consts"
)

func ResourceToOSSPath(resourceType, resource string) string {
	switch resourceType {
	case consts.Model:
		if strings.HasPrefix(resource, "user-") {
			return fmt.Sprintf(consts.OSSUserModelPath, resource)
		} else {
			return fmt.Sprintf(consts.OSSPublicModelPath, resource)
		}
	case consts.Dataset:
		if strings.HasPrefix(resource, "user-") {
			return fmt.Sprintf(consts.OSSUserDatasetPath, resource)
		} else {
			return fmt.Sprintf(consts.OSSPublicDatasetPath, resource)
		}
	}

	return ""
}

func ModelCRDName(ossPath string) string {
	return fmt.Sprintf("model-storage-%s", hash(ossPath))
}

func DatasetCRDName(ossPath string) string {
	return fmt.Sprintf("dataset-storage-%s", hash(ossPath))
}

func ModelPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-model-%s", hash(ossPath))
}

func DatasetPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-dataset-%s", hash(ossPath))
}

func ModelDownloadJobName(ossPath string) string {
	return fmt.Sprintf("download-model-%s", hash(ossPath))
}

func DatasetDownloadJobName(ossPath string) string {
	return fmt.Sprintf("download-dataset-%s", hash(ossPath))
}

func hash(data string) string {
	hash := sha1.New()
	hash.Write([]byte(data))
	hashed := hash.Sum(nil)
	return hex.EncodeToString(hashed)[:16]
}
