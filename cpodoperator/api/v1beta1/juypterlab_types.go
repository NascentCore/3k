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
type JuypterLabSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Image string `json:"image,omitempty"`

	CPUCount string `json:"cpuCount,omitempty"`

	Memory string `json:"memory,omitempty"`

	GPUCount int `json:"gpuCount,omitempty"`

	GPUProduct string `json:"gpuProduct,omitempty"`

	DataVolumeSize string `json:"dataVolumeSize,omitempty"`

	Models []Model `json:"model,omitempty"`
}

type Model struct {
	ModelStorage string `json:"modelStorage,omitempty"`
	MountPath    string `json:"mountPath,omitempty"`
}

type JupyterLabJobPhase string

const (
	JupyterLabJobPhasePending JupyterLabJobPhase = "Pending"
	JupyterLabJobPhaseRunning JupyterLabJobPhase = "Running"
	JupyterLabJobPhaseFailed  JupyterLabJobPhase = "Failed"
)

// JuypterLabStatus defines the observed state of JuypterLab
type JuypterLabStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Phase JupyterLabJobPhase `json:"phase,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

// JuypterLab is the Schema for the juypterlabs API
type JuypterLab struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   JuypterLabSpec   `json:"spec,omitempty"`
	Status JuypterLabStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// JuypterLabList contains a list of JuypterLab
type JuypterLabList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []JuypterLab `json:"items"`
}

func init() {
	SchemeBuilder.Register(&JuypterLab{}, &JuypterLabList{})
}
