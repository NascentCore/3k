package model

import (
    "context"
    "database/sql"
    
    "github.com/Masterminds/squirrel"
    "github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysUserModel = (*customSysUserModel)(nil)

type (
    // SysUserModel is an interface to be customized, add more methods here,
    // and implement the added methods in customSysUserModel.
    SysUserModel interface {
        sysUserModel
        AllFieldsBuilder() squirrel.SelectBuilder
        UpdateBuilder() squirrel.UpdateBuilder
        
        FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysUser, error)
        FindOneById(ctx context.Context, data *SysUser) (*SysUser, error)
        FindAll(ctx context.Context, orderBy string) ([]*SysUser, error)
        Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysUser, error)
        FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysUser, error)
        UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
        IsAdmin(ctx context.Context, userID int64) (bool, error)
    }
    
    customSysUserModel struct {
        *defaultSysUserModel
    }
)

// NewSysUserModel returns a model for the database table.
func NewSysUserModel(conn sqlx.SqlConn) SysUserModel {
    return &customSysUserModel{
        defaultSysUserModel: newSysUserModel(conn),
    }
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysUserModel) AllFieldsBuilder() squirrel.SelectBuilder {
    return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysUserModel) UpdateBuilder() squirrel.UpdateBuilder {
    return squirrel.Update(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysUserModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysUser, error) {
    selectBuilder = selectBuilder.Limit(1)
    query, args, err := selectBuilder.ToSql()
    if err != nil {
        return nil, err
    }
    
    var resp SysUser
    err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return &resp, nil
    default:
        return nil, err
    }
}

// FindOneById just like FindOneByQuery but use data.UserId as query condition
func (m *defaultSysUserModel) FindOneById(ctx context.Context, data *SysUser) (*SysUser, error) {
    return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("user_id = ?", data.UserId))
}

// FindAll returns all valid rows in the table
func (m *defaultSysUserModel) FindAll(ctx context.Context, orderBy string) ([]*SysUser, error) {
    selectBuilder := m.AllFieldsBuilder()
    if orderBy == "" {
        selectBuilder = selectBuilder.OrderBy("user_id DESC")
    } else {
        selectBuilder = selectBuilder.OrderBy(orderBy)
    }
    
    query, args, err := selectBuilder.ToSql()
    if err != nil {
        return nil, err
    }
    
    var resp []*SysUser
    err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return resp, nil
    default:
        return nil, err
    }
}

func (m *defaultSysUserModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysUser, error) {
    query, args, err := selectBuilder.ToSql()
    if err != nil {
        return nil, err
    }
    
    var resp []*SysUser
    err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return resp, nil
    default:
        return nil, err
    }
}

// FindPageListByPage -
func (m *defaultSysUserModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysUser, error) {
    if orderBy == "" {
        selectBuilder = selectBuilder.OrderBy("user_id DESC")
    } else {
        selectBuilder = selectBuilder.OrderBy(orderBy)
    }
    
    if page < 1 {
        page = 1
    }
    offset := (page - 1) * pageSize
    
    query, args, err := selectBuilder.Offset(uint64(offset)).Limit(uint64(pageSize)).ToSql()
    if err != nil {
        return nil, err
    }
    
    var resp []*SysUser
    err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return resp, nil
    default:
        return nil, err
    }
}

// UpdateColsByCond -
func (m *defaultSysUserModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
    query, args, err := updateBuilder.ToSql()
    if err != nil {
        return nil, err
    }
    
    return m.conn.ExecCtx(ctx, query, args...)
}

func (m *defaultSysUserModel) IsAdmin(ctx context.Context, userID int64) (bool, error) {
    user, err := m.FindOneById(ctx, &SysUser{UserId: userID})
    if err != nil {
        return false, err
    }
    
    return user.Admin > 0, nil
}
