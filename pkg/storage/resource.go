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
	case consts.Adapter:
		if strings.HasPrefix(resource, "user-") {
			return fmt.Sprintf(consts.OSSUserAdapterPath, resource)
		} else {
			return fmt.Sprintf(consts.OSSPublicAdapterPath, resource)
		}
	}

	return ""
}

func ResourceIsPublic(resource string) bool {
	if strings.HasPrefix(resource, "user-") {
		return false
	} else {
		return true
	}
}

func ModelCRDName(ossPath string) string {
	return fmt.Sprintf("model-storage-%s", hash(ossPath))
}

func DatasetCRDName(ossPath string) string {
	return fmt.Sprintf("dataset-storage-%s", hash(ossPath))
}

func AdapterCRDName(ossPath string) string {
	return fmt.Sprintf("adapter-storage-%s", hash(ossPath))
}

func ModelPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-model-%s", hash(ossPath))
}

func DatasetPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-dataset-%s", hash(ossPath))
}

func AdapterPVCName(ossPath string) string {
	return fmt.Sprintf("pvc-adapter-%s", hash(ossPath))
}

func ModelDownloadJobName(ossPath string) string {
	return fmt.Sprintf("download-model-%s", hash(ossPath))
}

func DatasetDownloadJobName(ossPath string) string {
	return fmt.Sprintf("download-dataset-%s", hash(ossPath))
}

func AdapterDownloadJobName(ossPath string) string {
	return fmt.Sprintf("download-adapter-%s", hash(ossPath))
}

func OssPathToOssURL(bucket, ossPath string) string {
	return fmt.Sprintf("oss://%s/%s", bucket, ossPath)
}

func hash(data string) string {
	hash := sha1.New()
	hash.Write([]byte(data))
	hashed := hash.Sum(nil)
	return hex.EncodeToString(hashed)[:16]
}

func ExtractTemplate(filename string) string {
	// Split the filename by "/"
	parts := strings.Split(filename, "/")
	// Get the last part of the split, which should be "<template>-<name>.<extension>"
	lastPart := parts[len(parts)-1]
	// Now split the last part by "-" and get the second to last element, which is assumed to be the template name
	nameParts := strings.Split(lastPart, "-")
	// Before extracting the name, we split the last part of nameParts by "." to remove the file extension
	templateWithExtension := nameParts[len(nameParts)-1]
	template := strings.Split(templateWithExtension, ".")[0]
	return template
}
