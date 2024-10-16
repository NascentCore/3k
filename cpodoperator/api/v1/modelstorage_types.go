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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ModelStorageSpec defines the desired state of ModelStorage
type ModelStorageSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	ModelType             string `json:"modeltype,omitempty"`
	ModelName             string `json:"modelname,omitempty"`
	PVC                   string `json:"pvc,omitempty"`
	Template              string `json:"template,omitempty"`
	ConvertTensorRTEngine bool   `json:"converttensortrengine,omitempty"`
	IsLoraAdapter         bool   `json:"isloraadapter,omitempty"`
	Category              string `json:"category,omitempty"`
}

type ModelCategory string

const (
	ModelCategoryChat      ModelCategory = "Chat"
	ModelCategoryEmbedding ModelCategory = "Embedding"
)

type ConvertTensorRTEngineStatus string

const (
	ConvertTensorRTEngineStatusConverting ConvertTensorRTEngineStatus = "Converting"
	ConvertTensorRTEngineStatusConverted  ConvertTensorRTEngineStatus = "Converted"
	ConvertTensorRTEngineStatusFailed     ConvertTensorRTEngineStatus = "Failed"
)

// ModelStorageStatus defines the observed state of ModelStorage
type ModelStorageStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Size                        int64                       `json:"size,omitempty"`
	Phase                       string                      `json:"phase,omitempty"`
	ConvertTensorRTEngineStatus ConvertTensorRTEngineStatus `json:"converttensortrenginestatus,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:storageversion
// +kubebuilder:printcolumn:name="ModelType",type=string,JSONPath=`.spec.modeltype`
// +kubebuilder:printcolumn:name="ModelName",type=string,JSONPath=`.spec.modelname`
// +kubebuilder:printcolumn:name="Template",type=string,JSONPath=`.spec.template`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Size",type=string,JSONPath=`.status.size`

// ModelStorage is the Schema for the modelstorages API
type ModelStorage struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ModelStorageSpec   `json:"spec,omitempty"`
	Status ModelStorageStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// ModelStorageList contains a list of ModelStorage
type ModelStorageList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ModelStorage `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ModelStorage{}, &ModelStorageList{})
}
