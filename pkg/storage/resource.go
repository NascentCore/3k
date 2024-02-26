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

func ModelCrdID(ossPath string) string {
	return fmt.Sprintf("model-storage-%s", hash(ossPath))
}

func DatasetCrdID(ossPath string) string {
	return fmt.Sprintf("dataset-storage-%s", hash(ossPath))
}

func hash(data string) string {
	hash := sha1.New()
	hash.Write([]byte(data))
	hashed := hash.Sum(nil)
	return hex.EncodeToString(hashed)[:16]
}
