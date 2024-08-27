package resource

import (
	"context"
	"encoding/json"
	"path"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/pkg/errors"
	"github.com/zeromicro/go-zero/core/logx"
)

type Manager struct {
	ctx context.Context
	logx.Logger
	svcCtx *svc.ServiceContext
}

func NewManager(svcCtx *svc.ServiceContext) *Manager {
	ctx := context.Background()
	return &Manager{
		ctx:    ctx,
		Logger: logx.WithContext(ctx),
		svcCtx: svcCtx,
	}
}

func (m *Manager) SyncOSS() {
	m.syncModels()
	m.syncDatasets()
	m.syncAdapter()
}

func (m *Manager) syncModels() {
	OssResourceModel := m.svcCtx.OssResourceModel

	// resource in db
	resourcesInDB, err := OssResourceModel.FindAll(m.ctx, "")
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		m.Errorf("OssResourceModel FindAll err: %v", err)
		return
	}

	resourceMap := make(map[string]bool)
	for _, resource := range resourcesInDB {
		resourceMap[resource.ResourceId] = true
	}

	var modelsToInsert, modelsToUpdate []model.SysOssResource

	// public models
	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(m.svcCtx.Config.OSS.Bucket, m.svcCtx.Config.OSS.PublicModelDir, 2)
	if err != nil {
		m.Errorf("SyncOSS err: %v", err)
		return
	}

	for dir, size := range dirs {
		canFinetune, _, err := storage.ExistFile(m.svcCtx.Config.OSS.Bucket,
			path.Join(dir, m.svcCtx.Config.OSS.FinetuneTagFile))
		if err != nil {
			m.Errorf("SyncOSS ExistFile err: %v", err)
			return
		}
		canInference, _, err := storage.ExistFile(m.svcCtx.Config.OSS.Bucket,
			path.Join(dir, m.svcCtx.Config.OSS.InferenceTagFile))
		if err != nil {
			m.Errorf("SyncOSS ExistFile err: %v", err)
			return
		}

		modelName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), m.svcCtx.Config.OSS.PublicModelDir)

		meta := model.OssResourceModelMeta{
			Template:     storage.ModelTemplate(m.svcCtx.Config.OSS.Bucket, modelName),
			Category:     consts.ModelCategoryChat, // oss上没有这个信息，默认就当做chat模型
			CanFinetune:  canFinetune,
			CanInference: canInference,
		}
		metaJson, err := json.Marshal(meta)
		if err != nil {
			m.Errorf("SyncOSS meta marshal err: %v", err)
			return
		}

		resourceID := storage.ModelCRDName(storage.ResourceToOSSPath(consts.Model, modelName))
		resource := model.SysOssResource{
			ResourceId:   resourceID,
			ResourceType: "model",
			ResourceName: modelName,
			ResourceSize: size,
			Public:       model.CachePublic,
			UserId:       "public",
			Meta:         string(metaJson),
		}

		_, ok := resourceMap[resourceID]
		if ok {
			modelsToUpdate = append(modelsToUpdate, resource)
		} else {
			modelsToInsert = append(modelsToInsert, resource)
		}
	}

	// insert
	err = m.Insert(modelsToInsert)
	if err != nil {
		m.Errorf("SyncOSS insert err: %v", err)
		return
	}

	// update
	err = m.Update(modelsToUpdate)
	if err != nil {
		m.Errorf("SyncOSS update err: %v", err)
		return
	}
}

func (m *Manager) syncDatasets() {
	OssResourceModel := m.svcCtx.OssResourceModel

	// resource in db
	resourcesInDB, err := OssResourceModel.FindAll(m.ctx, "")
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		m.Errorf("OssResourceModel FindAll err: %v", err)
		return
	}

	resourceMap := make(map[string]bool)
	for _, resource := range resourcesInDB {
		resourceMap[resource.ResourceId] = true
	}

	var datasetsToInsert, datasetsToUpdate []model.SysOssResource

	// public datasets
	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(m.svcCtx.Config.OSS.Bucket, m.svcCtx.Config.OSS.PublicDatasetDir, 2)
	if err != nil {
		m.Errorf("SyncOSS err: %v", err)
		return
	}

	for dir, size := range dirs {
		datasetName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), m.svcCtx.Config.OSS.PublicDatasetDir)

		resourceID := storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Dataset, datasetName))
		resource := model.SysOssResource{
			ResourceId:   resourceID,
			ResourceType: consts.Dataset,
			ResourceName: datasetName,
			ResourceSize: size,
			Public:       model.CachePublic,
			UserId:       "public",
			Meta:         "{}",
		}

		_, ok := resourceMap[resourceID]
		if ok {
			datasetsToUpdate = append(datasetsToUpdate, resource)
		} else {
			datasetsToInsert = append(datasetsToInsert, resource)
		}
	}

	// insert
	err = m.Insert(datasetsToInsert)
	if err != nil {
		m.Errorf("SyncOSS insert err: %v", err)
		return
	}

	// update
	err = m.Update(datasetsToUpdate)
	if err != nil {
		m.Errorf("SyncOSS update err: %v", err)
		return
	}
}

func (m *Manager) syncAdapter() {
	OssResourceModel := m.svcCtx.OssResourceModel

	// resource in db
	resourcesInDB, err := OssResourceModel.FindAll(m.ctx, "")
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		m.Errorf("OssResourceModel FindAll err: %v", err)
		return
	}

	resourceMap := make(map[string]bool)
	for _, resource := range resourcesInDB {
		resourceMap[resource.ResourceId] = true
	}

	var adaptersToInsert, adaptersToUpdate []model.SysOssResource

	// public adapters
	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(m.svcCtx.Config.OSS.Bucket, m.svcCtx.Config.OSS.PublicAdapterDir, 2)
	if err != nil {
		m.Errorf("SyncOSS err: %v", err)
		return
	}

	for dir, size := range dirs {
		adapterName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), m.svcCtx.Config.OSS.PublicAdapterDir)

		meta := model.OssResourceAdapterMeta{
			BaseModel: "default",
		}
		metaJson, err := json.Marshal(meta)
		if err != nil {
			m.Errorf("SyncOSS meta marshal err: %v", err)
			return
		}

		resourceID := storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Adapter, adapterName))
		resource := model.SysOssResource{
			ResourceId:   resourceID,
			ResourceType: consts.Adapter,
			ResourceName: adapterName,
			ResourceSize: size,
			Public:       model.CachePublic,
			UserId:       "public",
			Meta:         string(metaJson),
		}

		_, ok := resourceMap[resourceID]
		if ok {
			adaptersToUpdate = append(adaptersToUpdate, resource)
		} else {
			adaptersToInsert = append(adaptersToInsert, resource)
		}
	}

	// insert
	err = m.Insert(adaptersToInsert)
	if err != nil {
		m.Errorf("SyncOSS insert err: %v", err)
		return
	}

	// update
	err = m.Update(adaptersToUpdate)
	if err != nil {
		m.Errorf("SyncOSS update err: %v", err)
		return
	}
}

func (m *Manager) Insert(resources []model.SysOssResource) error {
	if len(resources) == 0 {
		return nil
	}

	insertBuilder := m.svcCtx.OssResourceModel.InsertBuilder().Columns(
		"resource_id",
		"resource_type",
		"resource_name",
		"resource_size",
		"public",
		"user_id",
		"meta",
	)
	for _, resource := range resources {
		insertBuilder = insertBuilder.Values(
			resource.ResourceId,
			resource.ResourceType,
			resource.ResourceName,
			resource.ResourceSize,
			resource.Public,
			resource.UserId,
			resource.Meta,
		)
	}

	query, args, err := insertBuilder.ToSql()
	if err != nil {
		m.Errorf("insertBuilder.ToSql err=%s", err)
		return err
	}

	_, err = m.svcCtx.DB.Exec(query, args...)
	if err != nil {
		m.Errorf("insert exec err=%s", err)
		return err
	}

	return nil
}

func (m *Manager) Update(resources []model.SysOssResource) error {
	for _, resource := range resources {
		updateBuilder := m.svcCtx.OssResourceModel.UpdateBuilder().Set(
			"resource_type", resource.ResourceType,
		).Set(
			"resource_name", resource.ResourceName,
		).Set(
			"resource_size", resource.ResourceSize,
		).Set(
			"public", resource.Public,
		).Set(
			"user_id", resource.UserId,
		).Set(
			"meta", resource.Meta,
		).Set(
			"updated_at", time.Now(),
		).Where(
			"resource_id = ?", resource.ResourceId,
		)

		query, args, err := updateBuilder.ToSql()
		if err != nil {
			m.Errorf("updateBuilder.ToSql err=%s", err)
			return err
		}

		_, err = m.svcCtx.DB.Exec(query, args...)
		if err != nil {
			m.Errorf("update exec err=%s", err)
			return err
		}
	}

	return nil
}
