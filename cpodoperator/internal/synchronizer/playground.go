package synchronizer

import (
	"context"
	"fmt"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/provider/litellm"
	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type Playground struct {
	kubeClient          client.Client
	litellm             *litellm.Litellm
	logger              logr.Logger
	playgroundNamespace string
	playgroundBaseURL   string
}

func NewPlayground(kubeClient client.Client, litellm *litellm.Litellm, playgroundBaseURL string, playgroundNamespace string, logger logr.Logger) *Playground {
	return &Playground{
		kubeClient:          kubeClient,
		litellm:             litellm,
		logger:              logger,
		playgroundBaseURL:   playgroundBaseURL,
		playgroundNamespace: playgroundNamespace,
	}
}

func (p *Playground) Start(ctx context.Context) {
	models, err := p.litellm.ListModels()
	if err != nil {
		p.logger.Error(err, "failed to list models")
		return
	}
	p.logger.Info("list models", "models", models.Data)
	var inferences v1beta1.InferenceList
	err = p.kubeClient.List(context.Background(), &inferences, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	}, client.InNamespace(p.playgroundNamespace))
	if err != nil {
		p.logger.Error(err, "failed to list inferences")
		return
	}
	p.logger.Info("list inferences", "inferences", inferences.Items)

	for _, inference := range inferences.Items {
		if !inference.Status.Ready {
			continue
		}
		modelName := inference.ObjectMeta.Annotations[v1beta1.CPodPreTrainModelReadableNameAnno]
		exists := false
		for _, m := range models.Data {
			if m.ModelName == modelName {
				exists = true
			}
		}
		if !exists {
			err := p.litellm.AddModel(litellm.Model{
				ModelName: modelName,
				LitellmParams: litellm.LitellmParams{
					APIBase: fmt.Sprintf("%s/inference/api/%s/v1", p.playgroundBaseURL, inference.ObjectMeta.Name),
					APIKey:  "test",
					Model:   fmt.Sprintf("%s/%s", "hosted_vllm", modelName),
				},
			})
			if err != nil {
				p.logger.Error(err, "failed to add model", "model", modelName)
			} else {
				p.logger.Info("added model", "model", modelName)
			}
		}
	}

	for _, model := range models.Data {
		if !model.ModelInfo.DBModel {
			continue
		}
		exists := false
		for _, inference := range inferences.Items {
			if inference.ObjectMeta.Annotations[v1beta1.CPodPreTrainModelReadableNameAnno] == model.ModelName {
				exists = true
			}
		}
		if !exists {
			err := p.litellm.RemoveModel(litellm.ModelRemoveRequest{ID: model.ModelInfo.ID})
			if err != nil {
				p.logger.Error(err, "failed to remove model", "model", model.ModelName)
			} else {
				p.logger.Info("removed model", "model", model.ModelName)
			}
		}
	}

}
