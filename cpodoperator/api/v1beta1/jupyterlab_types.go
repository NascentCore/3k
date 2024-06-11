/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// JuypterLabSpec defines the desired state of JuypterLab
type JupyterLabSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Image string `json:"image,omitempty"`

	CPUCount string `json:"cpuCount,omitempty"`

	Memory string `json:"memory,omitempty"`

	GPUCount int `json:"gpuCount,omitempty"`

	GPUProduct string `json:"gpuProduct,omitempty"`

	Replicas *int32 `json:"replicas"`

	DataVolumeSize string `json:"dataVolumeSize,omitempty"`

	Models []Model `json:"models,omitempty"`

	Datasets []Dataset `json:"datasets,omitempty"`
}

type Dataset struct {
	DatasetStorage  string `json:"datasetStorage,omitempty"`
	DatasetSize     int    `json:"datasetSize,omitempty"`
	DatasetIspublic bool   `json:"datasetIspublic,omitempty"`
	Name            string `json:"name,omitempty"`
	MountPath       string `json:"mountPath,omitempty"`
}

type Model struct {
	ModelStorage  string `json:"modelStorage,omitempty"`
	ModelIspublic bool   `json:"modelIspublic,omitempty"`
	ModelSize     int    `json:"modelSize,omitempty"`
	Name          string `json:"name,omitempty"`
	IsAdapter     bool   `json:"isAdapter,omitempty"`
	Template      string `json:"template,omitempty"`
	MountPath     string `json:"mountPath,omitempty"`
}

type JupyterLabJobPhase string

const (
	JupyterLabJobPhasePending JupyterLabJobPhase = "Pending" // 启动中
	JupyterLabJobPhasePaused  JupyterLabJobPhase = "Paused"  // 已暂停
	JupyterLabJobPhasePausing JupyterLabJobPhase = "Pausing" // 暂停中
	JupyterLabJobPhaseRunning JupyterLabJobPhase = "Running" //运行中
	JupyterLabJobPhaseFailed  JupyterLabJobPhase = "Failed"  // 失败
)

// JuypterLabStatus defines the observed state of JuypterLab
type JupyterLabStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Phase JupyterLabJobPhase `json:"phase,omitempty"`

	DataReady bool `json:"dataReady,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

// JupyterLab is the Schema for the jupyterlabs API
type JupyterLab struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   JupyterLabSpec   `json:"spec,omitempty"`
	Status JupyterLabStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// JupyterLabList contains a list of JupyterLab
type JupyterLabList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []JupyterLab `json:"items"`
}

func init() {
	SchemeBuilder.Register(&JupyterLab{}, &JupyterLabList{})
}
