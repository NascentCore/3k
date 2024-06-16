package util

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"strings"
)

const (
	Model                = "model"
	Dataset              = "dataset"
	Adapter              = "adapter"
	OSSPublicModelPath   = "models/public/%s"
	OSSUserModelPath     = "models/%s"
	OSSPublicDatasetPath = "datasets/public/%s"
	OSSUserDatasetPath   = "datasets/%s"
	OSSPublicAdapterPath = "adapters/public/%s"
	OSSUserAdapterPath   = "adapters/%s"
)

func ModelCRDName(ossPath string) string {
	return fmt.Sprintf("model-storage-%s", hash(ossPath))
}

func DatasetCRDName(ossPath string) string {
	return fmt.Sprintf("dataset-storage-%s", hash(ossPath))
}

func ModelPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-model-%s", hash(ossPath))
}

func AdapterPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-adapter-%s", hash(ossPath))
}

func DatasetPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-dataset-%s", hash(ossPath))
}

func ModelDownloadJobName(ossPath string) string {
	return fmt.Sprintf("download-model-%s", hash(ossPath))
}

func AdapterCRDName(ossPath string) string {
	return fmt.Sprintf("adapter-storage-%s", hash(ossPath))
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

func ResourceToOSSPath(resourceType, resource string) string {
	switch resourceType {
	case Model:
		if strings.HasPrefix(resource, "user-") {
			return fmt.Sprintf(OSSUserModelPath, resource)
		} else {
			return fmt.Sprintf(OSSPublicModelPath, resource)
		}
	case Dataset:
		if strings.HasPrefix(resource, "user-") {
			return fmt.Sprintf(OSSUserDatasetPath, resource)
		} else {
			return fmt.Sprintf(OSSPublicDatasetPath, resource)
		}
	case Adapter:
		if strings.HasPrefix(resource, "user-") {
			return fmt.Sprintf(OSSUserAdapterPath, resource)
		} else {
			return fmt.Sprintf(OSSPublicAdapterPath, resource)
		}
	}
	return ""
}
