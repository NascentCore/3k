package litellm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLitellm_ListModels(t *testing.T) {
	// 创建客户端
	client := &Litellm{
		BaseURL: "http://playground.llm.sxwl.ai:30005",
		APIKey:  "sk-1234",
	}

	// 执行测试
	models, err := client.ListModels()
	assert.NoError(t, err)
	assert.Len(t, models.Data, 1)
}

func TestLitellm_AddModel(t *testing.T) {
	// 创建客户端
	client := &Litellm{
		BaseURL: "http://playground.llm.sxwl.ai:30005",
		APIKey:  "sk-1234",
	}
	err := client.AddModel(Model{ModelName: "test-model", LitellmParams: LitellmParams{APIBase: "http://master.llm.sxwl.ai:30005/inference/api/infer-abf4f428-f7e9-4bef-9d7c-9ec10fab696a/v1", Model: "google/gemma-2b-it"}})
	assert.NoError(t, err)
}

func TestLitellm_RemoveModel(t *testing.T) {
	// 创建客户端
	client := &Litellm{
		BaseURL: "http://playground.llm.sxwl.ai:30005",
		APIKey:  "sk-1234",
	}
	err := client.RemoveModel(ModelRemoveRequest{ID: "54709c1a-a45e-4262-853a-e44937783c21"})
	assert.NoError(t, err)
}
