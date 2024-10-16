package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// YAMLResourceReconciler reconciles a YAMLResource object
type YAMLResourceReconciler struct {
	client.Client
	Scheme *runtime.Scheme
	Domain string
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=yamlresources,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=yamlresources/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=yamlresources/finalizers,verbs=update
//+kubebuilder:rbac:groups=*,resources=*,verbs=get;list;watch;create;update;patch;delete

func (r *YAMLResourceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Starting Reconcile", "request", req)

	yamlResource := &cpodv1beta1.YAMLResource{}
	if err := r.Get(ctx, req.NamespacedName, yamlResource); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("YAMLResource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get YAMLResource")
		return ctrl.Result{}, err
	}

	logger.Info("YAMLResource found", "yamlResource", yamlResource)

	// 如果资源的 phase 已经是 Running，直接退出，不再重复处理
	if yamlResource.Status.Phase == cpodv1beta1.YAMLResourcePhaseRunning {
		logger.Info("YAMLResource is already in Running phase, no further action required.")
		return ctrl.Result{}, nil
	}

	// 更新状态为 "Pending" 或 "Updating"
	if _, err := r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhasePending, "Processing YAML resources"); err != nil {
		logger.Error(err, "Failed to update status to Creating")
		return ctrl.Result{}, err
	}

	// 解析 yamlResource.Spec.Meta 并提取 env 内容
	var meta map[string]interface{}
	if err := json.Unmarshal([]byte(yamlResource.Spec.Meta), &meta); err != nil {
		logger.Error(err, "Failed to parse Meta JSON")
		return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to parse Meta JSON: "+err.Error())
	}

	// 提取 env 变量
	envMap, ok := meta["env"].(map[string]interface{})
	if !ok {
		logger.Info("No env content found in Meta, skipping environment variable injection")
		envMap = nil // 没有找到 env，后续不处理
	}

	// 使用 yaml.NewYAMLOrJSONDecoder 来处理多个资源
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(yamlResource.Spec.YAML)), 4096)
	for {
		var rawObj runtime.RawExtension
		if err := decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break // 已经处理完所有资源
			}
			logger.Error(err, "Failed to decode YAML")
			return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to decode YAML: "+err.Error())
		}

		obj, _, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
		if err != nil {
			logger.Error(err, "Failed to decode raw object")
			return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to decode raw object: "+err.Error())
		}

		unstructuredObj := obj.(*unstructured.Unstructured)

		// 设置资源的命名空间
		if unstructuredObj.GetNamespace() == "" {
			unstructuredObj.SetNamespace(yamlResource.Namespace)
		}

		// 添加 OwnerReferences
		ownerReference := metav1.OwnerReference{
			APIVersion:         yamlResource.APIVersion,
			Kind:               yamlResource.Kind,
			Name:               yamlResource.Name,
			UID:                yamlResource.UID,
			Controller:         ptr.To(true),
			BlockOwnerDeletion: ptr.To(true),
		}
		unstructuredObj.SetOwnerReferences([]metav1.OwnerReference{ownerReference})

		if unstructuredObj.GetKind() == "Ingress" {
			// 获取 spec.rules.host 并修改为 app_id + domain
			if rules, found, err := unstructured.NestedSlice(unstructuredObj.Object, "spec", "rules"); found && err == nil {
				for i, rule := range rules {
					ruleMap := rule.(map[string]interface{})
					if _, found := ruleMap["host"]; found {
						ruleMap["host"] = fmt.Sprintf("%v.%v", yamlResource.Name, r.Domain)
					}
					rules[i] = ruleMap
				}
				// 将修改后的规则设置回 unstructuredObj 中
				if err := unstructured.SetNestedSlice(unstructuredObj.Object, rules, "spec", "rules"); err != nil {
					return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to set Ingress host: "+err.Error())
				}
			} else {
				return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to get Ingress rules: "+err.Error())
			}
		}

		// 处理 StatefulSet 资源并添加环境变量
		if unstructuredObj.GetKind() == "StatefulSet" && envMap != nil {
			logger.Info("StatefulSet found, injecting environment variables from Meta")
			if err := r.addEnvToStatefulSet(unstructuredObj, envMap); err != nil {
				logger.Error(err, "Failed to inject environment variables into StatefulSet")
				return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to inject environment variables into StatefulSet: "+err.Error())
			}
		}

		// 创建或更新资源
		created, err := r.createOrUpdateResource(ctx, unstructuredObj)
		if err != nil {
			logger.Error(err, "Failed to create or update resource", "resource", unstructuredObj.GetKind())
			return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, fmt.Sprintf("Failed to %s resource %s: %v",
				map[bool]string{true: "create", false: "update"}[created],
				unstructuredObj.GetKind(), err))
		}

		logger.Info("Resource processed successfully", "resource", unstructuredObj.GetKind(), "operation", map[bool]string{true: "created", false: "updated"}[created])
	}

	if err := r.checkQanythingPodPortReady(ctx, yamlResource.Namespace); err != nil {
		logger.Error(err, "Pod qanything-0 is not ready or ports are not open")
		return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, err.Error())
	}

	// 更新 YAMLResource 状态
	return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseRunning, "All resources created/updated successfully")
}

// checkQanythingPodPortReady 循环检查名为 qanything-0 的 Pod 是否处于 Ready 状态以及端口是否就绪
func (r *YAMLResourceReconciler) checkQanythingPodPortReady(ctx context.Context, namespace string) error {
	logger := log.FromContext(ctx)
	pod := &unstructured.Unstructured{}
	pod.SetKind("Pod")
	pod.SetAPIVersion("v1")
	pod.SetNamespace(namespace)
	pod.SetName("qanything-0")

	// 设定最大重试次数和重试间隔
	maxRetries := 30
	retryInterval := 5 * time.Second

	for i := 0; i < maxRetries; i++ {
		// 获取名为 qanything-0 的 Pod
		if err := r.Get(ctx, client.ObjectKey{Name: "qanything-0", Namespace: namespace}, pod); err != nil {
			logger.Error(err, "Failed to get Pod qanything-0")
			time.Sleep(retryInterval)
			continue
		}

		// 检查 Pod 是否处于 Running 状态
		phase, found, err := unstructured.NestedString(pod.Object, "status", "phase")
		if err != nil || !found {
			logger.Error(err, "Failed to get status phase of Pod qanything-0")
			time.Sleep(retryInterval)
			continue
		}

		logger.Info("Pod status phase:", "phase", phase)

		if phase != "Running" {
			logger.Info("Pod qanything-0 is not in Running phase yet, retrying...")
			time.Sleep(retryInterval)
			continue
		}

		// 检查 Pod 的 Ready 条件是否为 True
		conditions, found, err := unstructured.NestedSlice(pod.Object, "status", "conditions")
		if err != nil || !found {
			logger.Error(err, "Failed to get conditions for Pod qanything-0")
			time.Sleep(retryInterval)
			continue
		}

		isReady := false
		for _, condition := range conditions {
			conditionMap := condition.(map[string]interface{})
			if conditionMap["type"] == "Ready" && conditionMap["status"] == "True" {
				isReady = true
				break
			}
		}

		if isReady {
			logger.Info("Pod qanything-0 is ready, checking ports...")

			// 获取 Pod 的 IP 地址
			podIP, found, err := unstructured.NestedString(pod.Object, "status", "podIP")
			if err != nil || !found {
				return fmt.Errorf("failed to get pod IP for qanything-0: %v", err)
			}

			// 检查端口是否已开放
			containers, found, err := unstructured.NestedSlice(pod.Object, "spec", "containers")
			if err != nil || !found {
				return fmt.Errorf("failed to get containers from Pod qanything-0: %v", err)
			}

			for _, container := range containers {
				containerMap := container.(map[string]interface{})
				ports, found, err := unstructured.NestedSlice(containerMap, "ports")
				if err != nil || !found {
					return fmt.Errorf("failed to get ports from container in Pod qanything-0: %v", err)
				}

				for _, port := range ports {
					portMap := port.(map[string]interface{})
					containerPort := portMap["containerPort"].(int64)

					// 尝试连接端口，检查是否监听
					if err := checkPortOpen(podIP, containerPort, 10, retryInterval); err != nil {
						logger.Error(err, "port is not open", "pod", "qanything-0", "port", containerPort)
						return fmt.Errorf("failed to check port open: %v", err)
					}

					logger.Info("Pod qanything-0 has port open:", "port", portMap["containerPort"])
				}
			}

			// Pod 端口检查完成，返回成功
			return nil
		}

		logger.Info("Pod qanything-0 is not ready yet, retrying...")
		time.Sleep(retryInterval)
	}

	// 超过最大重试次数，返回超时错误
	return fmt.Errorf("pod qanything-0 did not become ready after %d retries", maxRetries)
}

// checkPortOpen 尝试通过 TCP 连接到指定的 IP 和端口，检查端口是否监听
func checkPortOpen(ip string, port int64, retries int, retryInterval time.Duration) error {
	address := fmt.Sprintf("%s:%d", ip, port)
	for j := 0; j < retries; j++ {
		conn, err := net.DialTimeout("tcp", address, 2*time.Second) // 尝试 2 秒内连接
		if err != nil {
			if j >= retries-1 {
				// 达到最大重试次数后返回错误
				return fmt.Errorf("check port open timeout after %d retries", retries)
			}
			time.Sleep(retryInterval)
			continue
		}
		// 成功连接后关闭连接并退出循环
		conn.Close()
		break
	}
	return nil
}

func (r *YAMLResourceReconciler) addEnvToStatefulSet(unstructuredObj *unstructured.Unstructured, envMap map[string]interface{}) error {
	containers, found, err := unstructured.NestedSlice(unstructuredObj.Object, "spec", "template", "spec", "containers")
	if err != nil || !found {
		return fmt.Errorf("failed to get containers from StatefulSet: %v", err)
	}

	// 遍历 containers 并添加或更新 env 变量
	for i, container := range containers {
		containerMap := container.(map[string]interface{})

		// 获取现有的 env 列表
		env, found, err := unstructured.NestedSlice(containerMap, "env")
		if err != nil || !found {
			env = []interface{}{}
		}

		// 创建一个 map 来追踪现有的 env 变量
		existingEnvMap := make(map[string]int)
		for j, e := range env {
			envVar := e.(map[string]interface{})
			name := envVar["name"].(string)
			existingEnvMap[name] = j // 保存 env 变量的位置
		}

		// 将 envMap 中的 key-value 转换为 Kubernetes 环境变量格式
		for key, value := range envMap {
			envVar := map[string]interface{}{
				"name":  key,
				"value": fmt.Sprintf("%v", value),
			}

			// 如果环境变量已存在，进行更新
			if index, exists := existingEnvMap[key]; exists {
				env[index] = envVar
			} else {
				// 如果不存在，则追加新的环境变量
				env = append(env, envVar)
			}
		}

		// 更新 container 中的 env 列表
		if err := unstructured.SetNestedSlice(containerMap, env, "env"); err != nil {
			return fmt.Errorf("failed to set environment variables for container: %v", err)
		}

		containers[i] = containerMap
	}

	// 更新 StatefulSet 的 containers 列表
	if err := unstructured.SetNestedSlice(unstructuredObj.Object, containers, "spec", "template", "spec", "containers"); err != nil {
		return fmt.Errorf("failed to update containers in StatefulSet: %v", err)
	}

	return nil
}

func (r *YAMLResourceReconciler) createOrUpdateResource(ctx context.Context, obj *unstructured.Unstructured) (bool, error) {
	existingObj := &unstructured.Unstructured{}
	existingObj.SetGroupVersionKind(obj.GroupVersionKind())

	err := r.Get(ctx, client.ObjectKey{Namespace: obj.GetNamespace(), Name: obj.GetName()}, existingObj)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// 资源不存在,创建它
			return true, r.Create(ctx, obj)
		}
		return false, err
	}

	// 资源已存在,更新它
	obj.SetResourceVersion(existingObj.GetResourceVersion())
	return false, r.Update(ctx, obj)
}

func (r *YAMLResourceReconciler) updateStatus(ctx context.Context, yamlResource *cpodv1beta1.YAMLResource, phase cpodv1beta1.YAMLResourcePhase, message string) (ctrl.Result, error) {
	yamlResource.Status.Phase = phase
	yamlResource.Status.Message = message
	yamlResource.Status.LastSyncTime = &metav1.Time{Time: time.Now()}

	// 清除过时的 condition，将所有旧的 conditions 状态标记为 False
	for i := range yamlResource.Status.Conditions {
		yamlResource.Status.Conditions[i].Status = metav1.ConditionFalse
	}

	// 添加当前的状态 condition
	condition := metav1.Condition{
		Type:               string(phase),
		Status:             metav1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             string(phase),
		Message:            message,
	}
	meta.SetStatusCondition(&yamlResource.Status.Conditions, condition)

	if err := r.Status().Update(ctx, yamlResource); err != nil {
		log.FromContext(ctx).Error(err, "Failed to update YAMLResource status")
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *YAMLResourceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.YAMLResource{}).
		Complete(r)
}
