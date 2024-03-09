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

func (m *Model) ConstructCommandArgs(finetuneName string, hyperparameters, configs []string) string {
	baseParam := []string{
		"accelerate",
		"launch",
		"--num_processes=1",
		"src/train_bash.py",
		"--do_train=True",
		"--model_name_or_path=/data/model",
		"--dataset=dataset",
		"--dataset_dir=/data/dataset",
		"--output_dir=/data/ckpt",
		"--stage=sft",
		"--finetuning_type=lora",
		"--lr_scheduler_type=cosine",
		"--overwrite_cache=True",
		"--gradient_accumulation_steps=4",
		"--logging_steps=10",
		"--save_steps=1000",
		"--plot_loss=True",
		"--fp16=True",
		"--report_to=tensorboard",
		"--logging_dir=/logs/" + finetuneName,
		fmt.Sprintf("--template=%v", m.Template),
		fmt.Sprintf("--lora_target=%v", m.LoRATarget),
	}

	trainStrings := append(append(baseParam, hyperparameters...), configs...)
	var res string
	for _, v := range trainStrings {
		res += v + " "
	}

	return res + fmt.Sprintf(" && python src/export_model.py --model_name_or_path /data/model --adapter_name_or_path /data/ckpt --template %v --finetuning_type lora --export_dir=/data/save", m.Template)
}

var SupportModels = []Model{
	{
		Name:             "ChatGLM3",
		ModelStorageName: "model-storage-13e253bdc2c04565",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v1",
		Template:         "alpaca",
		LoRATarget:       "query_key_value",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "LLaMA-2-7B",
		ModelStorageName: "model-storage-dd6b73224fa6ab8c",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v1",
		Template:         "alpaca",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "Baichuan2-7B",
		ModelStorageName: "model-storage-e5d919757ed2d808",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v1",
		Template:         "alpaca",
		LoRATarget:       "W_pack",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "Ziya-LLaMA-13B-v1",
		ModelStorageName: "model-storage-bce7b3f87937e7d3",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v1",
		Template:         "alpaca",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "Gemma-2B-it",
		ModelStorageName: "model-storage-89b0f24c4991ed83",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v1",
		Template:         "gemma",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "Mixtral-7B",
		ModelStorageName: "model-storage-926fa37a09dd4724",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v1",
		Template:         "mistral",
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
