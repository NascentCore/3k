package finetune

import "fmt"

type Model struct {
	Name             string
	ModelStorageName string
	Image            string
	Template         string
	LoRATarget       string
	// targetmodelsize represents the size of the target model, it's unit is MB.
	Targetmodelsize int
	RequireGPUType  string
}

func (m *Model) ConstructCommand(hyperparameters, configs []string) []string {
	baseParam := []string{
		"accelerate",
		"launch",
		"--num_processes=1",
		"src/train_bash.py",
		"--do_train=True",
		"--model_name_or_path=/data/model",
		"--dataset=dataset",
		"--dataset_dir=/data/dataset",
		"--output_dir=/data/save",
		"--stage=sft",
		"--finetuning_type=lora",
		"--lr_scheduler_type=cosine",
		"--overwrite_cache=True",
		"--gradient_accumulation_steps=4",
		"--logging_steps=10",
		"--save_steps=1000",
		"--plot_loss=True",
		"--fp16=True",
		fmt.Sprintf("--template=%v", m.Template),
		fmt.Sprintf("--lora_target=%v", m.LoRATarget),
	}

	return append(append(baseParam, hyperparameters...), configs...)
}

var SupportModels = []Model{
	{
		Name:             "ChatGLM3",
		ModelStorageName: "chatglm-6b",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:latest",
		Template:         "alpaca",
		LoRATarget:       "query_key_value",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "LLaMA-2-7B",
		ModelStorageName: "llama-2-7b-hf",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:latest",
		Template:         "alpaca",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "Baichuan2-7B",
		ModelStorageName: "model-storage-d69aabdc8e017114",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:latest",
		Template:         "alpaca",
		LoRATarget:       "W_pack",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "Ziya-LLaMA-13B-v1",
		ModelStorageName: "model-storage-bf51e08c872c690e",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:latest",
		Template:         "alpaca",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
}

func CheckModelWhetherSupport(modelName string) *Model {
	for _, model := range SupportModels {
		if model.Name == modelName {
			return &model
		}
	}
	return nil
}

func ConvertHyperParameter(hyperParameters map[string]string) map[string]string {
	result := make(map[string]string)
	for k, v := range hyperParameters {
		switch k {
		case "n_epochs":
			result["num_train_epochs"] = v
		case "batch_size":
			result["per_device_train_batch_size"] = v
		case "learning_rate":
			result["learning_rate"] = v
		}
	}
	return result
}
