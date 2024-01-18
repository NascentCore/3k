package record

import (
	"sxwl/3k/cmd/downloader/internal/consts"
	"sxwl/3k/cmd/downloader/internal/record/crd"
	"sxwl/3k/cmd/downloader/internal/record/none"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type Recorder interface {
	Check() error
	Begin() error
	Fail() error
	Complete() error
}

type Config struct {
	Group     string // "cpod.cpod"
	Version   string // "v1"
	Plural    string // "modelstorages"
	Name      string // "example-modelstorage"
	Namespace string // "cpod"
	Depth     uint   // 1
}

func NewRecorder(recorderType string, c Config) Recorder {
	switch recorderType {
	case consts.CRD:
		return crd.NewRecorder(schema.GroupVersionResource{
			Group:    c.Group,
			Version:  c.Version,
			Resource: c.Plural,
		}, c.Name, c.Namespace)
	default:
		return none.NewRecorder()
	}
}
