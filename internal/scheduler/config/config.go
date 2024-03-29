package config

import (
	"github.com/zeromicro/go-zero/rest"
)

type Config struct {
	rest.RestConf
	DB struct {
		DataSource string
	} `json:"-"`
	OSS struct {
		Bucket            string
		Endpoint          string
		PublicModelDir    string
		PublicDatasetDir  string
		UserModelDir      string
		UserModelPrefix   string
		UserDatasetDir    string
		UserDatasetPrefix string
		FinetuneTagFile   string
		InferenceTagFile  string
		LocalMode         bool
	} `json:"OSS"`
	BannedCpod map[string]string `json:"BannedCpod"`
	OSSAccess  struct {
		UploadAccessID  string
		UploadAccessKey string
		AdminAccessID   string
		AdminAccessKey  string
	} `json:"-"`
	FinetuneModel map[string]string `json:"FinetuneModel"`
	Inference     struct {
		UrlFormat string
	} `json:"Inference"`
}
