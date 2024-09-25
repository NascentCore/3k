package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// YAMLResourceSpec defines the desired state of YAMLResource
type YAMLResourceSpec struct {
	YAML    string `json:"yaml"`
	AppName string `json:"appName"`
	AppID   string `json:"appId"`
	UserID  string `json:"userId"`
}

// YAMLResourceStatus defines the observed state of YAMLResource
type YAMLResourceStatus struct {
	// Phase 表示 YAMLResource 的当前状态
	Phase YAMLResourcePhase `json:"phase,omitempty"`
	// Message 提供关于当前状态的额外信息
	Message string `json:"message,omitempty"`
	// LastSyncTime 是最后一次成功同步的时间
	LastSyncTime *metav1.Time `json:"lastSyncTime,omitempty"`
	// Conditions 代表 YAMLResource 的当前服务状态
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// YAMLResourcePhase is a label for the condition of a YAMLResource at the current time
type YAMLResourcePhase string

const (
	YAMLResourcePhasePending YAMLResourcePhase = "Pending"
	YAMLResourcePhaseRunning YAMLResourcePhase = "Running"
	YAMLResourcePhaseFailed  YAMLResourcePhase = "Failed"
)

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

// YAMLResource is the Schema for the yamlresources API
type YAMLResource struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   YAMLResourceSpec   `json:"spec,omitempty"`
	Status YAMLResourceStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// YAMLResourceList contains a list of YAMLResource
type YAMLResourceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []YAMLResource `json:"items"`
}

func init() {
	SchemeBuilder.Register(&YAMLResource{}, &YAMLResourceList{})
}
