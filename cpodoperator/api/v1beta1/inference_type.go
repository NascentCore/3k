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
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	duckv1 "knative.dev/pkg/apis/duck/v1"
)

// InferenceSpec defines the desired state of Inference
type InferenceSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Predictor defines the model serving spec
	// +required
	Predictor kservev1beta1.PredictorSpec `json:"predictor"`
}

// InferenceStatus defines the observed state of Inference
type InferenceStatus struct {
	Ready bool `json:"ready,omitempty"`

	URL *string `json:"url,omitempty"`

	Conditions duckv1.Conditions `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// Inference is the Schema for the Inferences API
type Inference struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferenceSpec   `json:"spec,omitempty"`
	Status InferenceStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// InferenceList contains a list of Inference
type InferenceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Inference `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Inference{}, &InferenceList{})
}
