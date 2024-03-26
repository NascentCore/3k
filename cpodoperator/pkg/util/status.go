package util

import (
	"github.com/NascentCore/cpodoperator/api/v1beta1"

	tov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func IsFinshed(status v1beta1.CPodJobStatus) bool {
	return IsSucceeded(status) || IsFailed(status)
}

// IsSucceeded checks if the given CPodJobStatus is in a succeeded state.
// It returns true if the status has a condition with the type JobSucceeded and the condition is true.
func IsSucceeded(status v1beta1.CPodJobStatus) bool {
	return isStatusConditionTrue(status, v1beta1.JobSucceeded)
}

func IsModelUploaded(status v1beta1.CPodJobStatus) bool {
	return isStatusConditionTrue(status, v1beta1.JobModelUploaded)
}

// IsFailed checks if the given CPodJobStatus is in a failed state.
// It returns true if the status has a condition with the type JobFailed and the condition is true.
func IsFailed(status v1beta1.CPodJobStatus) bool {
	return isStatusConditionTrue(status, v1beta1.JobFailed)
}

func isStatusConditionTrue(status v1beta1.CPodJobStatus, condType v1beta1.JobConditionType) bool {
	for _, condition := range status.Conditions {
		if condition.Type == condType {
			return condition.Status == v1.ConditionTrue
		}
	}
	return false
}

// newCondition creates a new job condition.
func newCondition(
	conditionType v1beta1.JobConditionType,
	conditionStatus v1.ConditionStatus,
	reason, message string,
) v1beta1.JobCondition {
	return v1beta1.JobCondition{
		Type:               conditionType,
		Status:             conditionStatus,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func UpdateJobConditions(
	jobStatus *v1beta1.CPodJobStatus,
	conditionType v1beta1.JobConditionType,
	conditionStatus v1.ConditionStatus,
	reason, message string,
) {
	condition := newCondition(conditionType, conditionStatus, reason, message)
	setCondition(jobStatus, condition)
}

// setCondition updates the job to include the provided condition.
// If the condition already exists, it will be updated.
func setCondition(jobStatus *v1beta1.CPodJobStatus, condition v1beta1.JobCondition) {
	// implementation of setCondition
	if IsFailed(*jobStatus) {
		return
	}

	currentCond := GetCondition(*jobStatus, condition.Type)

	// Do nothing if condidtion doesn't chage
	if currentCond != nil && currentCond.Status == condition.Status && currentCond.Reason == condition.Reason && currentCond.Message == condition.Message {
		return
	}

	// Do not update lastTransitionTime if the status of the condition doesn't change.
	if currentCond != nil && currentCond.Status == condition.Status {
		condition.LastTransitionTime = currentCond.LastTransitionTime
	}

	newConditions := filterOutCondition(jobStatus.Conditions, condition.Type)
	jobStatus.Conditions = append(newConditions, condition)

}

func GetCondition(status v1beta1.CPodJobStatus, condType v1beta1.JobConditionType) *v1beta1.JobCondition {
	for _, condition := range status.Conditions {
		if condition.Type == condType {
			return &condition
		}
	}
	return nil
}

// filterOutCondition returns a new slice of JobCondition without conditions with the provided type.
func filterOutCondition(conditions []v1beta1.JobCondition, condType v1beta1.JobConditionType) []v1beta1.JobCondition {
	var newConditions []v1beta1.JobCondition
	for _, condition := range conditions {
		if condition.Type == condType {
			continue
		}
		// Set the running condition to false if the job is finished(succeeded or failed).
		if (condType == v1beta1.JobFailed || condType == v1beta1.JobSucceeded) && condition.Type == v1beta1.JobRunning {
			condition.Status = v1.ConditionFalse
		}

		if condType == v1beta1.JobModelUploaded && condition.Type == v1beta1.JobModelUploading {
			continue
		}

		newConditions = append(newConditions, condition)
	}
	return newConditions
}

// GetCondition returns the baseJob condition with the provided type.
func GetBaseJobCondition(status tov1.JobStatus, condType tov1.JobConditionType) *tov1.JobCondition {
	for _, condition := range status.Conditions {
		if condition.Type == condType {
			return &condition
		}
	}
	return nil
}
