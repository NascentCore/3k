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

// DataSetStorageSpec defines the desired state of DataSetStorage
type DataSetStorageSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	DatasetType string `json:"datasettype,omitempty"`
	DatasetName string `json:"datasetname,omitempty"`
	PVC         string `json:"pvc,omitempty"`
}

// DataSetStorageStatus defines the observed state of DataSetStorage
type DataSetStorageStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Phase string `json:"phase,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="DatasetType",type=string,JSONPath=`.spec.datasettype`
// +kubebuilder:printcolumn:name="DatasetName",type=string,JSONPath=`.spec.datasetname`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`

// DataSetStorage is the Schema for the datasetstorages API
type DataSetStorage struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DataSetStorageSpec   `json:"spec,omitempty"`
	Status DataSetStorageStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// DataSetStorageList contains a list of DataSetStorage
type DataSetStorageList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DataSetStorage `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DataSetStorage{}, &DataSetStorageList{})
}
