package db

import "gorm.io/gorm"

type CpodResources struct {
	Cpods []CpodResource `json:"cpods"`
}

type CpodResource struct {
	gorm.Model
	Id       int     `json:"id,omitempty"`
	GroupId  string  `json:"group_id,omitempty"`
	CpodId   string  `json:"cpod_id"`
	GpuTotal float32 `json:"gpu_total"`
	GpuUsed  float32 `json:"gpu_used"`
	GpuFree  float32 `json:"gpu_free"`
}

func (r *CpodResource) CreateResource(db *gorm.DB) (string, error) {
	results := db.Table("cpod_resource").Where("cpod_id =?", r.CpodId)
	if results.Error != nil {
		if results.Error == gorm.ErrRecordNotFound {
			results := db.Table("cpod_resource").Create(r)
			if results.Error != nil {
				return results.Error.Error(), results.Error
			}
		}
	} else {
		results := db.Table("cpod_resource").Updates(r)
		if results.Error != nil {
			return results.Error.Error(), results.Error
		}
	}
	return "", nil
}

func (r *CpodResource) UpdateResource(db *gorm.DB) (string, error) {
	results := db.Table("cpod_resource").Updates(r)
	if results.Error != nil {
		return results.Error.Error(), results.Error
	}
	return "", nil
}
