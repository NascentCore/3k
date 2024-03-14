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

	finetunepkg "github.com/NascentCore/cpodoperator/pkg/finetune"
	"github.com/NascentCore/cpodoperator/pkg/util"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
)

type FineTuneOption struct {
	GPUProduct string
}

// FineTuneReconciler reconciles a FineTune object
type FineTuneReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	Option *FineTuneOption
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=finetunes,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=finetunes/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=finetunes/finalizers,verbs=update
//+kubebuilder:rbac:groups=cpod.cpod,resources=cpodjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=cpodjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=cpodjobs/finalizers,verbs=update
//+kubebuilder:rbac:groups=cpod.cpod,resources=modelstorages,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=datasetstorages,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the FineTune object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.16.3/pkg/reconcile
func (r *FineTuneReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	finetune := &cpodv1beta1.FineTune{}
	if err := r.Get(ctx, req.NamespacedName, finetune); client.IgnoreNotFound(err) != nil {
		logger.Error(err, "unabel to fetch finetune")
		return ctrl.Result{}, err
	}

	if finetune.DeletionTimestamp != nil {
		return ctrl.Result{}, nil
	}

	if finetune.Status.Phase == cpodv1beta1.PhaseFailed || finetune.Status.Phase == cpodv1beta1.PhaseSucceeded {
		return ctrl.Result{}, nil
	}

	validateErr, recoverableErr := r.validateFineTune(ctx, finetune)
	if validateErr != nil {
		logger.Error(validateErr, "validate finetune error")
		return ctrl.Result{}, nil
	}

	if recoverableErr != nil {
		return ctrl.Result{}, recoverableErr
	}

	cpodjob := &cpodv1beta1.CPodJob{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: finetune.Namespace, Name: r.CPodJobName(finetune)}, cpodjob); err != nil {
		if apierrors.IsNotFound(err) {
			var modelConfig *finetunepkg.Model
			if modelConfig = finetunepkg.CheckModelWhetherSupport(finetune.Spec.Model); modelConfig == nil {
				logger.Error(fmt.Errorf("model not support"), "model not support")
				return ctrl.Result{}, nil
			}
			commandArg := modelConfig.ConstructCommandArgs(finetune.Name, ConvertParamsMap(finetunepkg.ConvertHyperParameter(finetune.Spec.HyperParameters)), ConvertParamsMap(finetune.Spec.Config))

			finetunCPodJob := cpodv1beta1.CPodJob{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: finetune.Namespace,
					Name:      r.CPodJobName(finetune),
					OwnerReferences: []metav1.OwnerReference{
						r.generateOwnerRefInference(finetune),
					},
				},
				Spec: cpodv1beta1.CPodJobSpec{
					Image:                 modelConfig.Image,
					JobType:               "pytorch",
					DatasetName:           finetune.Spec.DatasetStorage,
					DatasetPath:           "/data/dataset/custom",
					GPUType:               r.Option.GPUProduct,
					GPURequiredPerReplica: 1,
					ModelSavePath:         "/data/save",
					ModelSaveVolumeSize:   int32(modelConfig.Targetmodelsize),
					PretrainModelName:     modelConfig.ModelStorageName,
					PretrainModelPath:     "/data/model",
					CKPTPath:              "/data/ckpt",
					CKPTVolumeSize:        int32(modelConfig.Targetmodelsize),
					Command:               []string{"/bin/bash", "-c"},
					Args:                  []string{commandArg},
					UploadModel:           finetune.Spec.Upload,
				},
			}

			if err := r.Create(ctx, &finetunCPodJob); err != nil {
				logger.Error(err, "create cpodjob error")
				return ctrl.Result{}, err
			}
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{}, err
	}

	if util.IsFinshed(cpodjob.Status) {
		if util.IsSucceeded(cpodjob.Status) {
			finetune.Status.Phase = cpodv1beta1.PhaseSucceeded
		} else {
			finetune.Status.Phase = cpodv1beta1.PhaseFailed
			finetune.Status.FailureMessage = util.GetCondition(cpodjob.Status, cpodv1beta1.JobFailed).Message
		}
		if err := r.Client.Status().Update(ctx, finetune); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	} else {
		if finetune.Status.Phase != cpodv1beta1.PhaseRunning {
			finetune.Status.Phase = cpodv1beta1.PhaseRunning
			if err := r.Client.Status().Update(ctx, finetune); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{Requeue: true}, nil
	}
}

// getAvailableGPUType get available gpu type from k8s cluster
func (r *FineTuneReconciler) getAvailableGPUType(ctx context.Context) ([]string, error) {
	nodes := &v1.NodeList{}
	if err := r.Client.List(ctx, nodes); err != nil {
		return nil, err
	}

	gpuTypes := []string{}
	for _, node := range nodes.Items {
		if product, ok := node.Labels["nvidia.com/gpu.product"]; ok {
			gpuTypes = append(gpuTypes, product)
		}
	}

	return gpuTypes, nil
}

func (r *FineTuneReconciler) validateFineTune(ctx context.Context, finetune *cpodv1beta1.FineTune) (validateErr, RecoverableError error) {
	if finetune.Spec.Model == "" {
		return fmt.Errorf("model is required"), nil
	}

	if finetune.Spec.DatasetStorage == "" {
		return fmt.Errorf("dataset is required"), nil
	}

	if finetunepkg.CheckModelWhetherSupport(finetune.Spec.Model) == nil {
		return fmt.Errorf("model is not support"), nil
	}

	if err := r.Client.Get(ctx, types.NamespacedName{Namespace: finetune.Namespace, Name: finetune.Spec.DatasetStorage}, &cpodv1.DataSetStorage{}); err != nil {
		if apierrors.IsNotFound(err) {
			return fmt.Errorf("dataset storage not found"), nil
		}
		return nil, err
	}

	return nil, nil
}

func (r *FineTuneReconciler) generateOwnerRefInference(finetune *cpodv1beta1.FineTune) metav1.OwnerReference {
	yes := true
	return metav1.OwnerReference{
		APIVersion:         cpodv1beta1.GroupVersion.String(),
		Kind:               "FineTune",
		Name:               finetune.Name,
		UID:                finetune.GetUID(),
		Controller:         &yes,
		BlockOwnerDeletion: &yes,
	}
}

func (r *FineTuneReconciler) CPodJobName(finetune *cpodv1beta1.FineTune) string {
	return finetune.Name + "-cpodjob"
}

func (r *FineTuneReconciler) createCPodJob(ctx context.Context, finetune *cpodv1beta1.FineTune) error {
	return nil

}

// SetupWithManager sets up the controller with the Manager.
func (r *FineTuneReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.FineTune{}).
		Complete(r)
}

func ConvertParamsMap(params map[string]string) []string {
	result := make([]string, len(params))
	for key, value := range params {
		param := fmt.Sprintf("--%v=%v", key, value)
		result = append(result, param)
	}
	return result
}
