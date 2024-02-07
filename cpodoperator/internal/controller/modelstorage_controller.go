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
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
)

type ModelStorageOption struct {
	DownloaderImage      string
	TensorRTConvertImage string
	StorageClassName     string
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

	if modelStorage.Status.ConvertTensorRTEngineStatus == cpodv1.ConvertTensorRTEngineStatusConverted || modelStorage.Status.ConvertTensorRTEngineStatus == cpodv1.ConvertTensorRTEngineStatusFailed {
		return ctrl.Result{}, nil
	}

	convertJob := batchv1.Job{}
	if err := r.Client.Get(ctx, client.ObjectKey{Namespace: modelStorage.Namespace, Name: "convert-job-" + modelStorage.Name}, &convertJob); err != nil {
		if apierrors.IsNotFound(err) {
			createErr := r.createConvertJob(ctx, &modelStorage)
			if createErr != nil {
				return ctrl.Result{}, createErr
			}
			return ctrl.Result{Requeue: true}, nil
		}
	}

	if convertJob.Status.Succeeded == 1 {
		if err := r.updateConvertModelstorageStatus(ctx, &modelStorage); err != nil {
			return ctrl.Result{}, err
		}

		modelStorage.Status.ConvertTensorRTEngineStatus = cpodv1.ConvertTensorRTEngineStatusConverted
	} else {
		modelStorage.Status.ConvertTensorRTEngineStatus = cpodv1.ConvertTensorRTEngineStatusConverting
	}

	if err := r.Client.Status().Update(ctx, &modelStorage); err != nil {
		return ctrl.Result{}, err
	}

	if modelStorage.Status.ConvertTensorRTEngineStatus == cpodv1.ConvertTensorRTEngineStatusConverting {
		return ctrl.Result{Requeue: true}, nil
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ModelStorageReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1.ModelStorage{}).
		Complete(r)
}

func (r *ModelStorageReconciler) createConvertJob(ctx context.Context, modelStorage *cpodv1.ModelStorage) error {
	jobName := "convert-job-" + modelStorage.Name

	destPVC := corev1.PersistentVolumeClaim{}
	if err := r.Client.Get(ctx, client.ObjectKey{Namespace: modelStorage.Namespace, Name: modelStorage.Spec.PVC + "-dest"}, &destPVC); err != nil {
		if apierrors.IsNotFound(err) {
			srcPvc := corev1.PersistentVolumeClaim{}
			if err := r.Client.Get(ctx, client.ObjectKey{Namespace: modelStorage.Namespace, Name: modelStorage.Spec.PVC}, &srcPvc); err != nil {
				ctrl.Log.Error(err, "failed to get src pvc", "pvc", modelStorage.Spec.PVC)
				return err
			}

			if createErr := r.Client.Create(ctx, &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       modelStorage.Namespace,
					Name:            modelStorage.Spec.PVC + "-dest",
					OwnerReferences: []metav1.OwnerReference{},
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes:      []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
					Resources:        srcPvc.Spec.Resources,
					StorageClassName: srcPvc.Spec.StorageClassName,
					VolumeMode:       srcPvc.Spec.VolumeMode,
				},
			}); createErr != nil {
				return createErr
			}
			return err
		}
		return err
	}

	if destPVC.Status.Phase != corev1.ClaimBound {
		return fmt.Errorf("dest pvc is not bound")
	}

	destModelstorage := cpodv1.ModelStorage{}
	if err := r.Client.Get(ctx, client.ObjectKey{Namespace: modelStorage.Namespace, Name: modelStorage.Name + "-engine"}, &destModelstorage); err != nil {
		if apierrors.IsNotFound(err) {
			if createErr := r.Client.Create(ctx, &cpodv1.ModelStorage{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: modelStorage.Namespace,
					Name:      modelStorage.Name + "-engine",
					OwnerReferences: []metav1.OwnerReference{
						r.generateOwnerReference(modelStorage),
					},
				},
				Spec: cpodv1.ModelStorageSpec{
					ModelType: modelStorage.Spec.ModelType,
					ModelName: modelStorage.Spec.ModelName,
					PVC:       destPVC.Name,
				},
			}); createErr != nil {
				return createErr
			}
			return err
		}
		return err
	}

	convertJob := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: modelStorage.Namespace,
			Name:      jobName,
			Labels: map[string]string{
				v1beta1.ModelStorageLabel: modelStorage.Name,
			},
			OwnerReferences: []metav1.OwnerReference{
				r.generateOwnerReference(modelStorage),
			},
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "generater",
							Image:   r.Option.TensorRTConvertImage,
							Command: []string{"bash", "/opt/tritonserver/build_engine.sh", "/src", "/dest"},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "srcvolume",
									MountPath: "/src",
								},
								{
									Name:      "destvolume",
									MountPath: "/dest",
								},
							},
						},
					},
					RestartPolicy: corev1.RestartPolicyOnFailure,
					Volumes: []corev1.Volume{
						{
							Name: "srcvolume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: modelStorage.Spec.PVC,
								}},
						},
						{
							Name: "destvolume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: destPVC.Name,
								}},
						},
					},
				},
			},
		},
	}

	if err := client.IgnoreAlreadyExists(r.Client.Create(ctx, &convertJob)); err != nil {
		return err
	}
	return nil
}

func (r *ModelStorageReconciler) generateOwnerReference(modelStorage *cpodv1.ModelStorage) metav1.OwnerReference {
	yes := true
	return metav1.OwnerReference{
		APIVersion:         v1beta1.GroupVersion.String(),
		Kind:               "ModelStorage",
		Name:               modelStorage.Name,
		UID:                modelStorage.GetUID(),
		Controller:         &yes,
		BlockOwnerDeletion: &yes,
	}
}

func (r *ModelStorageReconciler) updateConvertModelstorageStatus(ctx context.Context, modelStorage *cpodv1.ModelStorage) error {
	convertModelstorage := cpodv1.ModelStorage{}
	if err := r.Client.Get(ctx, client.ObjectKey{Namespace: modelStorage.Namespace, Name: modelStorage.Name + "-engine"}, &convertModelstorage); err != nil {
		ctrl.Log.Error(err, "failed to get convert modelstorage", "modelstorage", modelStorage.Name)
		return err
	}
	convertModelstorage.Status.Phase = "done"
	return r.Client.Status().Update(ctx, &convertModelstorage)
}
