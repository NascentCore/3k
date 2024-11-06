package config

import (
	"sxwl/3k/pkg/email"

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
		PublicAdapterDir  string
		UserModelDir      string
		UserModelPrefix   string
		UserDatasetDir    string
		UserDatasetPrefix string
		UserAdapterDir    string
		UserAdapterPrefix string
		FinetuneTagFile   string
		InferenceTagFile  string
		SyncInterval      int64
		SyncCron          string
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
	K8S           struct {
		BaseApi string // k8s管理
		BaseUrl string // 组件入口
		AppUrl  string // 大模型应用URL
	} `json:"-"`
	Auth struct {
		Secret string
	} `json:"-"`
	Email email.Config `json:"-"`
	Rsa   struct {
		PrivateKey string
		PublicKey  string
	} `json:"-"`
	Billing struct {
		InitBalance float64
		CronBilling bool
		CronBalance bool
	} `json:"Billing"`
	DingTalk struct {
		AppKey    string
		AppSecret string
	} `json:"-"`
	ResourceLoad struct {
		On        bool
		CacheDir  string
		CacheSize int64
	} `json:"-"`
}
