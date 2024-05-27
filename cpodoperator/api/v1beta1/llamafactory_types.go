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

// LlamaFactorySpec defines the desired state of LlamaFactory
type LlamaFactorySpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Image string `json:"image,omitempty"`

	GPUCount int `json:"gpuCount,omitempty"`

	Memory string `json:"memory,omitempty"`
}

type LlamaFactoryPhase string

const (
	LlamaFactoryPending LlamaFactoryPhase = "Pending"
	LlamaFactoryRunning LlamaFactoryPhase = "Running"
	LlamaFactoryFailed  LlamaFactoryPhase = "Failed"
)

// LlamaFactoryStatus defines the observed state of LlamaFactory
type LlamaFactoryStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Phase LlamaFactoryPhase `json:"phase,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

// LlamaFactory is the Schema for the llamafactories API
type LlamaFactory struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   LlamaFactorySpec   `json:"spec,omitempty"`
	Status LlamaFactoryStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// LlamaFactoryList contains a list of LlamaFactory
type LlamaFactoryList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []LlamaFactory `json:"items"`
}

func init() {
	SchemeBuilder.Register(&LlamaFactory{}, &LlamaFactoryList{})
}
