package crd

import (
	"sxwl/3k/cmd/downloader/internal/consts"
	"sxwl/3k/cmd/downloader/internal/errors"
	clientgo "sxwl/3k/pkg/cluster/client-go"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type Recorder struct {
	gvr       schema.GroupVersionResource
	name      string
	namespace string
}

func NewRecorder(gvr schema.GroupVersionResource, name string, namespace string) *Recorder {
	clientgo.InitClient()
	return &Recorder{gvr: gvr, name: name, namespace: namespace}
}

func (r *Recorder) Check() error {
	// check crd exists
	dataMap, err := clientgo.GetObjectData(r.namespace, r.gvr.Group, r.gvr.Version, r.gvr.Resource, r.name)
	if err != nil {
		return err
	}

	_, ok := dataMap["status"]
	if !ok {
		return nil // crd without status
	}

	statusMap, ok := dataMap["status"].(map[string]interface{})
	if !ok {
		return errors.ErrCrdDataType
	}

	_, ok = statusMap["phase"]
	if !ok {
		return nil // crd without status.phase
	}

	phase, ok := statusMap["phase"].(string)
	if !ok {
		return errors.ErrCrdDataType
	}

	switch phase {
	case consts.PhaseFail:
		return nil
	case consts.PhaseDownloading:
		return errors.ErrJobDownloading
	case consts.PhaseComplete:
		return errors.ErrJobComplete
	default:
		return errors.ErrUnsupportedPhase
	}
}

func (r *Recorder) Begin() error {
	return clientgo.UpdateCRDStatus(r.gvr, r.namespace, r.name, consts.Phase, consts.PhaseDownloading)
}

func (r *Recorder) Fail() error {
	return clientgo.UpdateCRDStatus(r.gvr, r.namespace, r.name, consts.Phase, consts.PhaseFail)
}

func (r *Recorder) Complete() error {
	return clientgo.UpdateCRDStatus(r.gvr, r.namespace, r.name, consts.Phase, consts.PhaseComplete)
}

func (r *Recorder) Name() string {
	return consts.CRD
}
