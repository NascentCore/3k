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

	"github.com/NascentCore/cpodoperator/internal/synchronizer"
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
	"github.com/NascentCore/cpodoperator/api/v1beta1"
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

	gpuCount := finetune.Spec.GPUCount
	if gpuCount == 0 {
		gpuCount = 1
	}

	gpuProduct := finetune.Spec.GPUProduct
	if gpuProduct == "" {
		gpuProduct = r.Option.GPUProduct
	}

	cpodjob := &cpodv1beta1.CPodJob{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: finetune.Namespace, Name: r.CPodJobName(finetune)}, cpodjob); err != nil {
		if apierrors.IsNotFound(err) {
			var modelConfig *finetunepkg.Model
			if modelConfig = finetunepkg.CheckModelWhetherSupport(finetune.Spec.Model); modelConfig == nil {
				logger.Error(fmt.Errorf("model not support"), "model not support")
				return ctrl.Result{}, nil
			}
			commandArg := modelConfig.ConstructCommandArgs(finetune.Name, gpuCount, ConvertParamsMap(finetunepkg.ConvertHyperParameter(finetune.Spec.HyperParameters)), ConvertParamsMap(finetune.Spec.Config))

			if err := r.CopyPublicModelStorage(ctx, modelConfig.ModelStorageName, finetune); err != nil {
				logger.Error(err, "copy public model storage error")
				return ctrl.Result{}, err
			}

			finetunCPodJob := cpodv1beta1.CPodJob{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:   finetune.Namespace,
					Name:        r.CPodJobName(finetune),
					Labels:      map[string]string{},
					Annotations: finetune.Annotations,
					OwnerReferences: []metav1.OwnerReference{
						r.generateOwnerRefInference(finetune),
					},
				},
				Spec: cpodv1beta1.CPodJobSpec{
					Image:                 modelConfig.Image,
					JobType:               "pytorch",
					DatasetName:           finetune.Spec.DatasetStorage,
					DatasetPath:           "/data/dataset/custom",
					GPUType:               gpuProduct,
					GPURequiredPerReplica: gpuCount,
					ModelSavePath:         "/data/save",
					ModelSaveVolumeSize:   int32(modelConfig.Targetmodelsize),
					PretrainModelName:     modelConfig.ModelStorageName + v1beta1.CPodPublicStorageSuffix,
					PretrainModelPath:     "/data/model",
					CKPTPath:              "/data/ckpt",
					CKPTVolumeSize:        int32(modelConfig.Targetmodelsize),
					Command:               []string{"/bin/bash", "-c"},
					Args:                  []string{commandArg},
					UploadModel:           finetune.Spec.Upload,
				},
			}

			if userID, ok := finetune.Labels[v1beta1.CPodUserIDLabel]; ok {
				finetunCPodJob.Labels[v1beta1.CPodUserIDLabel] = userID
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
		if util.IsSucceeded(cpodjob.Status) && (!finetune.Spec.Upload || (finetune.Spec.Upload && util.IsModelUploaded(cpodjob.Status))) {
			finetune.Status.Phase = cpodv1beta1.PhaseSucceeded
			modelstorageName := cpodjob.Name + "-modelsavestorage"
			if userId, ok := finetune.Labels[v1beta1.CPodUserIDLabel]; ok {
				modelstorageName = synchronizer.ModelCRDName(fmt.Sprintf(synchronizer.OSSUserModelPath, userId+"/"+finetune.Name))
			}
			finetune.Status.ModelStorage = modelstorageName
		} else if util.IsFailed(cpodjob.Status) {
			finetune.Status.Phase = cpodv1beta1.PhaseFailed
			finetune.Status.FailureMessage = util.GetCondition(cpodjob.Status, cpodv1beta1.JobFailed).Message
		} else {
			return ctrl.Result{Requeue: true}, nil
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

func (r *FineTuneReconciler) CopyPublicModelStorage(ctx context.Context, publicModelStorageName string, finetune *cpodv1beta1.FineTune) error {
	modelStorage := &cpodv1.ModelStorage{}
	modelStorageName := publicModelStorageName + v1beta1.CPodPublicStorageSuffix
	publicModelStorage := &cpodv1.ModelStorage{}
	if err := r.Client.Get(ctx, types.NamespacedName{Namespace: finetune.Namespace, Name: modelStorageName}, modelStorage); err != nil {
		if apierrors.IsNotFound(err) {
			// if model storage not found, create a new one
			if err := r.Client.Get(ctx, types.NamespacedName{Namespace: v1beta1.CPodPublicNamespace, Name: publicModelStorageName}, publicModelStorage); err != nil {
				if apierrors.IsNotFound(err) {
					// 公共数据集不存在
					return fmt.Errorf("public model storage not found")
				}
				return err
			}
			// 创建用户命名空间的公共数据集的拷贝
			var publicDsPVC v1.PersistentVolumeClaim
			if err := r.Client.Get(ctx, types.NamespacedName{Namespace: v1beta1.CPodPublicNamespace, Name: publicModelStorage.Spec.PVC}, &publicDsPVC); err != nil {
				return fmt.Errorf("cannot find public model storage pvc")
			}
			var publicDsPV v1.PersistentVolume
			if err := r.Client.Get(ctx, types.NamespacedName{Namespace: v1beta1.CPodPublicNamespace, Name: publicDsPVC.Spec.VolumeName}, &publicDsPV); err != nil {
				return fmt.Errorf("cannt find public model storage pv")
			}
			// 创建pv
			pvCopy := publicDsPV.DeepCopy()
			pvCopy.Name = pvCopy.Name + "-" + finetune.Namespace
			pvCopy.ResourceVersion = ""
			pvCopy.Spec.CSI.VolumeHandle = pvCopy.Spec.CSI.VolumeHandle + "-" + finetune.Namespace
			pvCopy.UID = ""
			if err := r.Client.Create(ctx, pvCopy); err != nil && !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create pv")
			}
			// 创建pvc
			pvcCopy := publicDsPVC.DeepCopy()
			pvcCopy.Name = pvCopy.Name + "-" + finetune.Namespace
			pvcCopy.Namespace = finetune.Namespace
			pvcCopy.Spec.VolumeName = pvCopy.Name
			pvcCopy.ResourceVersion = ""
			pvcCopy.UID = ""
			if err := r.Client.Create(ctx, pvcCopy); err != nil && !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create pvc")
			}
			// 创建modelstorage
			modelStorageCopy := publicModelStorage.DeepCopy()
			modelStorageCopy.Name = publicModelStorage.Name + v1beta1.CPodPublicStorageSuffix
			modelStorageCopy.Namespace = finetune.Namespace
			modelStorageCopy.Spec.PVC = pvcCopy.Name
			modelStorageCopy.ResourceVersion = ""
			modelStorageCopy.UID = ""
			if modelStorageCopy.Labels == nil {
				modelStorageCopy.Labels = map[string]string{}
			}
			modelStorageCopy.Labels[v1beta1.CPODStorageCopyLable] = "true"
			if err := r.Client.Create(ctx, modelStorageCopy); err != nil {
				return err
			}
			// TODO: update modelstorage status
			modelStorageCopy.Status.Phase = "done"
			if err := r.Client.Status().Update(ctx, modelStorageCopy); err != nil {
				return fmt.Errorf("failed to update model storage status")
			}
			return nil
		}
		return err
	}

	return nil
}
