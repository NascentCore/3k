package modelhub

import (
	"errors"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
)

var ErrModelNotFound = errors.New("Model not found")

var UnsupportedModelHub = errors.New("unsupported modelhub")

// ModeHubInterface define model hub api
type ModelHubInterface interface {
	// ModelInformation get model information
	ModelInformation(modelID string, revision string) (interface{}, error)

	ModelSize(modelID string, revision string) (int64, error)

	ModelGitPath(modelID string) string
}

func ModelHubClient(modeType v1beta1.ModelType) (ModelHubInterface, error) {
	if modeType == v1beta1.ModelTypeModelscope {
		return ModelscopeHub, nil
	}
	return nil, UnsupportedModelHub
}
