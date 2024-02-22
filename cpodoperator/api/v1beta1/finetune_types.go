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

// FineTuneSpec defines the desired state of FineTune
type FineTuneSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Foo is an example field of FineTune. Edit finetune_types.go to remove/update
	Model           string            `json:"model,omitempty"`
	DatasetStorage  string            `json:"dataset,omitempty"`
	HyperParameters map[string]string `json:"hyperParameters,omitempty"`
	Config          map[string]string `json:"config,omitempty"`
}

type Phase string

const (
	PhasePending   Phase = "Pending"
	PhaseRunning   Phase = "Running"
	PhaseFailed    Phase = "Invalid"
	PhaseSucceeded Phase = "Succeeded"
)

// FineTuneStatus defines the observed state of FineTune
type FineTuneStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Phase Phase `json:"phase,omitempty"`

	FailureMessage string `json:"failureMessage,omitempty"`

	ModelStorage string `json:"modelStorage,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.model`
// +kubebuilder:printcolumn:name="Dataset",type=string,JSONPath=`.spec.dataset`
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"

// FineTune is the Schema for the finetunes API
type FineTune struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   FineTuneSpec   `json:"spec,omitempty"`
	Status FineTuneStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// FineTuneList contains a list of FineTune
type FineTuneList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []FineTune `json:"items"`
}

func init() {
	SchemeBuilder.Register(&FineTune{}, &FineTuneList{})
}
