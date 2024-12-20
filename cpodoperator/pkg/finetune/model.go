package finetune

import (
	"fmt"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
)

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

func (m *Model) ConstructCommandArgs(finetuneName string, finetuningType v1beta1.FinetuneType, gpuCount int32, merge bool, hyperparameters, configs []string) string {

	trainStrings := []string{
		"accelerate",
		"launch",
	}

	if gpuCount > 1 {
		accelerateConfig := []string{
			"--use_deepspeed",
			fmt.Sprintf("--num_processes=%v", gpuCount),
			"--zero_stage=3",
			"--zero3_save_16bit_model=true",
			"--offload_optimizer_device=cpu",
			"--offload_param_device=cpu",
			"--zero3_init_flag=true",
			"--gradient_accumulation_steps=4",
			"--machine_rank=0",
			"--num_machines=1",
			"--main_training_function=main",
			"--rdzv_backend=static",
		}

		trainStrings = append(trainStrings, accelerateConfig...)
	}

	outputDir := "/data/save"
	if finetuningType == "" {
		finetuningType = v1beta1.FinetuneTypeLora
	} else if finetuningType == v1beta1.FintuneTypeFull {
		finetuningType = "full"
	}

	baseParam := []string{
		"src/train.py",
		"--do_train=True",
		"--model_name_or_path=/data/model",
		"--dataset=dataset",
		"--dataset_dir=/data/dataset",
		fmt.Sprintf("--output_dir=%v", outputDir),
		"--stage=sft",
		fmt.Sprintf("--finetuning_type=%v", finetuningType),
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
	trainStrings = append(trainStrings, baseParam...)
	trainStrings = append(trainStrings, hyperparameters...)
	trainStrings = append(trainStrings, configs...)

	var res string
	for _, v := range trainStrings {
		res += v + " "
	}

	if merge && finetuningType != "full" {
		return res + fmt.Sprintf(" && llamafactory-cli export --model_name_or_path /data/model --adapter_name_or_path /data/ckpt --template %v --finetuning_type lora --export_dir=/data/save", m.Template) + fmt.Sprintf(" && cp /data/model/*.md /data/save/ 2>/dev/null || : ")
	}

	return res
}

var SupportModels = []Model{
	{
		Name:             "ZhipuAI/chatglm3-6b",
		ModelStorageName: "model-storage-10e872cd960e38cb",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v12",
		Template:         "alpaca",
		LoRATarget:       "query_key_value",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "meta-llama/Llama-2-7b",
		ModelStorageName: "model-storage-6deacca0bbad927d",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v12",
		Template:         "alpaca",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "baichuan-inc/Baichuan2-7B-Chat",
		ModelStorageName: "model-storage-cfa5be686c53f1a2",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "alpaca",
		LoRATarget:       "W_pack",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "IDEA-CCNL/Ziya-LLaMA-13B-v1",
		ModelStorageName: "model-storage-3f7baf1d50fdab32",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "alpaca",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "google/gemma-2b-it",
		ModelStorageName: "model-storage-0ce92f029254ff34",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "gemma",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "mistralai/Mistral-7B-v0.1",
		ModelStorageName: "model-storage-e306a7d8b79c7e8f",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "mistral",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "mistralai/Mistral-7B-Instruct-v0.1",
		ModelStorageName: "model-storage-9b268c705d2aafee",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "mistral",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "mistralai/Mixtral-8x7B-Instruct-v0.1",
		ModelStorageName: "model-storage-57e056b59249bceb",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "mistral",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  102400,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "meta-llama/Meta-Llama-3-8B-Instruct",
		ModelStorageName: "model-storage-f5b8963792864b06",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "llama3",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  102400,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "meta-llama/Meta-Llama-3-8B-Instruct",
		ModelStorageName: "model-storage-f5b8963792864b06",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha6",
		Template:         "llama3",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  143360,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
	{
		Name:             "LLM-Research/Llama-3.2-1B-Instruct",
		ModelStorageName: "model-storage-91f1b10f1fad84d0",
		Image:            "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:finetune-v1alpha7",
		Template:         "llama3",
		LoRATarget:       "q_proj,v_proj",
		Targetmodelsize:  30720,
		RequireGPUType:   "NVIDIA-GeForce-RTX-3090",
	},
}

func CheckModelWhetherSupport(supportModels []Model, modelName string) *Model {
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
