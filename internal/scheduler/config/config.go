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
	} `json:"OSS"`
	OSSAccess struct {
		UploadAccessID  string
		UploadAccessKey string
		AdminAccessID   string
		AdminAccessKey  string
	} `json:"-"`
	FinetuneModel map[string]string `json:"FinetuneModel"`
	Inference     map[string]struct {
		UrlFormat string `json:"url_format"`
	} `json:"inference"`
}
