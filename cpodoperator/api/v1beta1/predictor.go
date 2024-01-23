package v1beta1

import (
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	"github.com/kserve/kserve/pkg/constants"
)

type Predictor kservev1beta1.PredictorSpec

func (p *Predictor) SetStorageURI(url string) {
	if p.SKLearn != nil {
		p.SKLearn.StorageURI = &url
	}
	if p.PyTorch != nil {
		p.PyTorch.StorageURI = &url
	}
	if p.Triton != nil {
		p.Triton.StorageURI = &url
	}
	if p.SKLearn != nil {
		p.SKLearn.StorageURI = &url
	}
	if p.Tensorflow != nil {
		p.Tensorflow.StorageURI = &url
	}
	if p.ONNX != nil {
		p.ONNX.StorageURI = &url
	}
	if p.PMML != nil {
		p.PMML.StorageURI = &url
	}
	if p.LightGBM != nil {
		p.LightGBM.StorageURI = &url
	}
	if p.Paddle != nil {
		p.Paddle.StorageURI = &url
	}
	if p.Model != nil {
		p.Model.StorageURI = &url
	}

	if len(p.PodSpec.Containers) != 0 {
		for i, container := range p.PodSpec.Containers {
			if container.Name == constants.InferenceServiceContainerName {
				for j, envVar := range container.Env {
					if envVar.Name == constants.CustomSpecStorageUriEnvVarKey {
						p.PodSpec.Containers[i].Env[j].Value = url
					}
				}
				break
			}
		}
	}
}
