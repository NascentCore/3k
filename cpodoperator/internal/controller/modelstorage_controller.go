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
	"errors"
	"fmt"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/modelhub"
)

type ModelStorageOption struct {
	DownloaderImage  string
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

	modelStorage := v1beta1.ModelStorage{}
	if err := r.Client.Get(ctx, req.NamespacedName, &modelStorage); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	modelHubClient, err := modelhub.ModelHubClient(modelStorage.Spec.Type)
	if err != nil {
		return ctrl.Result{}, err
	}

	if modelStorage.Status.TargetPVCName == "" {
		size, err := modelHubClient.ModelSize(modelStorage.Spec.Name, "")
		if err != nil {
			if errors.Is(err, modelhub.ErrModelNotFound) {
				modelStorageDeepCopy := modelStorage.DeepCopy()
				errMessage := modelhub.ErrModelNotFound.Error()
				modelStorageDeepCopy.Status.FailureMessage = &errMessage
				if err := r.Client.Status().Update(ctx, modelStorageDeepCopy); err != nil {
					return ctrl.Result{}, err
				}
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, fmt.Errorf("failed to get model %v size: %v", req.NamespacedName, err)
		}

		modelStorage.Status.FailureMessage = nil

		// pvcName := base64.StdEncoding.EncodeToString([]byte(modelStorage.Spec.Type + "-" + cpodv1beta1.ModelType(modelStorage.Spec.Name)))
		pvcName := "pvc-" + modelStorage.Name
		yes := true
		pvc := &corev1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: modelStorage.Namespace,
				Name:      pvcName,
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
			Spec: corev1.PersistentVolumeClaimSpec{
				AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceStorage: *resource.NewQuantity(size*1024*1024*1024, resource.BinarySI),
					},
				},
				StorageClassName: &r.Option.StorageClassName,
			},
		}

		if client.IgnoreAlreadyExists(r.Client.Create(ctx, pvc)); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to create pvc: %v", err)
		}

		modelStorage.Status.TargetPVCName = pvcName

		if err := r.Client.Status().Update(ctx, &modelStorage); err != nil {
			return ctrl.Result{}, nil
		}

	}

	if modelStorage.Status.TargetDownloadJobName == "" {
		// create download job
		// jobName := base64.StdEncoding.EncodeToString([]byte(modelStorage.Spec.Type + "-" + cpodv1beta1.ModelType(modelStorage.Spec.Name)))
		jobName := "job-" + modelStorage.Name
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
								Name:    "downloader",
								Image:   r.Option.DownloaderImage,
								Command: []string{"git", "-s", modelHubClient.ModelGitPath(modelStorage.Spec.Name)},
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
										ClaimName: modelStorage.Status.TargetPVCName,
									}},
							},
						},
					},
				},
			},
		}

		if err := client.IgnoreAlreadyExists(r.Client.Create(ctx, &downloadJob)); err != nil {
			return ctrl.Result{}, err
		}

		modelStorage.Status.TargetDownloadJobName = jobName
		modelStorage.Status.Phase = v1beta1.ModelStoragePhaseSyncing

		if err := r.Client.Status().Update(ctx, &modelStorage); err != nil {
			return ctrl.Result{
				RequeueAfter: 10 * time.Second,
			}, nil
		}

	}

	var job batchv1.Job
	if err := r.Client.Get(ctx, types.NamespacedName{Namespace: modelStorage.Namespace, Name: modelStorage.Status.TargetDownloadJobName}, &job); err != nil {
		if apierrors.IsNotFound(err) {
			modelStorageDeepCopy := modelStorage.DeepCopy()
			modelStorageDeepCopy.Status.TargetDownloadJobName = ""
			if err := r.Client.Status().Update(ctx, modelStorageDeepCopy); err != nil {
				return ctrl.Result{}, nil
			}
		}
		// TODO: log
		logger.Error(err, "failed to get modelstorage", "name", modelStorage.Status.TargetDownloadJobName)
		return ctrl.Result{}, err
	}

	if job.Status.Succeeded == 1 {
		modelStorage.Status.Phase = v1beta1.ModelStoragePhaseDone
		if err := r.Client.Status().Update(ctx, &modelStorage); err != nil {
			return ctrl.Result{}, nil
		}
	} else {
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
