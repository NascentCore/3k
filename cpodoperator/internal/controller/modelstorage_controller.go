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

package controller

import (
	"context"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
)

type ModelStorageOption struct {
	DownloaderImage  string
	GeneraterImages  string
	StorageClassName string
}

// ModelStorageReconciler reconciles a ModelStorage object
type ModelStorageReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	Option *ModelStorageOption
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=modelstorages,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=modelstorages/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=modelstorages/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the ModelStorage object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.16.3/pkg/reconcile
func (r *ModelStorageReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	modelStorage := cpodv1.ModelStorage{}
	if err := r.Client.Get(ctx, req.NamespacedName, &modelStorage); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !modelStorage.Spec.ConvertTensorRTEngine {
		return ctrl.Result{}, nil
	}

	if modelStorage.Status.Phase != "done" {
		return ctrl.Result{Requeue: true}, nil
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ModelStorageReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.ModelStorage{}).
		Complete(r)
}

func (r *ModelStorageReconciler) createGenerateJob(ctx context.Context, modelStorage *cpodv1.ModelStorage) error {
	jobName := "generate-job-" + modelStorage.Name
	yes := true
	downloadJob := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: modelStorage.Namespace,
			Name:      jobName,
			Labels: map[string]string{
				v1beta1.ModelStorageLabel: modelStorage.Name,
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion:         v1beta1.GroupVersion.String(),
					Kind:               "ModelStorage",
					Name:               modelStorage.Name,
					UID:                modelStorage.GetUID(),
					Controller:         &yes,
					BlockOwnerDeletion: &yes,
				},
			},
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "generater",
							Image:   r.Option.GeneraterImages,
							Command: []string{},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "data-volume",
									MountPath: "/data",
								},
							},
						},
					},
					RestartPolicy: corev1.RestartPolicyOnFailure,
					Volumes: []corev1.Volume{
						{
							Name: "data-volume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: modelStorage.Spec.PVC,
								}},
						},
					},
				},
			},
		},
	}

	if err := client.IgnoreAlreadyExists(r.Client.Create(ctx, &downloadJob)); err != nil {
		return err
	}

	modelStorage.Status.TargetDownloadJobName = jobName
	modelStorage.Status.Phase = v1beta1.ModelStoragePhaseSyncing

	if err := r.Client.Status().Update(ctx, modelStorage); err != nil {
		return err
	}

	return nil
}
