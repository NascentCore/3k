package {{.pkg}}
{{if .withCache}}
import (
    "context"
    "database/sql"
    "time"
    
    "github.com/Masterminds/squirrel"
    "github.com/zeromicro/go-zero/core/stores/cache"
    "github.com/zeromicro/go-zero/core/stores/sqlx"
)
{{else}}
import (
    "context"
    "database/sql"
    "time"
    
    "github.com/Masterminds/squirrel"
    "github.com/zeromicro/go-zero/core/stores/sqlx"
)
{{end}}
var _ {{.upperStartCamelObject}}Model = (*custom{{.upperStartCamelObject}}Model)(nil)

type (
	// {{.upperStartCamelObject}}Model is an interface to be customized, add more methods here,
	// and implement the added methods in custom{{.upperStartCamelObject}}Model.
	{{.upperStartCamelObject}}Model interface {
		{{.lowerStartCamelObject}}Model
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder
		{{if .withCache}}

        {{else}}
		DeleteSoft(ctx context.Context, data *{{.upperStartCamelObject}}) error
		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*{{.upperStartCamelObject}}, error)
		FindOneById(ctx context.Context, data *{{.upperStartCamelObject}}) (*{{.upperStartCamelObject}}, error)
		FindAll(ctx context.Context, orderBy string) ([]*{{.upperStartCamelObject}}, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*{{.upperStartCamelObject}}, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*{{.upperStartCamelObject}}, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
        {{end}}
	}

	custom{{.upperStartCamelObject}}Model struct {
		*default{{.upperStartCamelObject}}Model
	}
)

// New{{.upperStartCamelObject}}Model returns a model for the database table.
func New{{.upperStartCamelObject}}Model(conn sqlx.SqlConn{{if .withCache}}, c cache.CacheConf, opts ...cache.Option{{end}}) {{.upperStartCamelObject}}Model {
	return &custom{{.upperStartCamelObject}}Model{
		default{{.upperStartCamelObject}}Model: new{{.upperStartCamelObject}}Model(conn{{if .withCache}}, c, opts...{{end}}),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *default{{.upperStartCamelObject}}Model) AllFieldsBuilder() squirrel.SelectBuilder {
    return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *default{{.upperStartCamelObject}}Model) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// DeleteSoft set deleted_at with CURRENT_TIMESTAMP
func (m *default{{.upperStartCamelObject}}Model) DeleteSoft(ctx context.Context, data *{{.upperStartCamelObject}}) error {
    builder := squirrel.Update(m.table)
    builder = builder.Set("deleted_at", sql.NullTime{
        Time:  time.Now(),
        Valid: true,
    })
    builder = builder.Where("id = ?", data.Id)
    query, args, err := builder.ToSql()
    if err != nil {
        return err
    }

    if _, err := m.conn.ExecCtx(ctx, query, args...); err != nil {
        return err
    }
    return nil
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *default{{.upperStartCamelObject}}Model) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*{{.upperStartCamelObject}}, error) {
    selectBuilder = selectBuilder.Where("deleted_at is null").Limit(1)
    query, args, err := selectBuilder.ToSql()
    if err != nil {
        return nil, err
    }

    var resp {{.upperStartCamelObject}}
    err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return &resp, nil
    default:
        return nil, err
    }
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *default{{.upperStartCamelObject}}Model) FindOneById(ctx context.Context, data *{{.upperStartCamelObject}}) (*{{.upperStartCamelObject}}, error) {
    return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *default{{.upperStartCamelObject}}Model) FindAll(ctx context.Context, orderBy string) ([]*{{.upperStartCamelObject}}, error) {
    selectBuilder := m.AllFieldsBuilder()
    if orderBy == "" {
        selectBuilder = selectBuilder.OrderBy("id DESC")
    } else {
        selectBuilder = selectBuilder.OrderBy(orderBy)
    }

    query, args, err := selectBuilder.Where("deleted_at is null").ToSql()
    if err != nil {
        return nil, err
    }

    var resp []*{{.upperStartCamelObject}}
    err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return resp, nil
    default:
        return nil, err
    }
}

// Find returns all valid rows matched in the table
func (m *default{{.upperStartCamelObject}}Model) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*{{.upperStartCamelObject}}, error) {
    query, args, err := selectBuilder.Where("deleted_at is null").ToSql()
    if err != nil {
        return nil, err
    }

    var resp []*{{.upperStartCamelObject}}
    err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return resp, nil
    default:
        return nil, err
    }
}

// FindPageListByPage -
func (m *default{{.upperStartCamelObject}}Model) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*{{.upperStartCamelObject}}, error) {
    if orderBy == "" {
        selectBuilder = selectBuilder.OrderBy("id DESC")
    } else {
        selectBuilder = selectBuilder.OrderBy(orderBy)
    }

    if page < 1 {
        page = 1
    }
    offset := (page - 1) * pageSize

    query, args, err := selectBuilder.Where("deleted_at is null").Offset(uint64(offset)).Limit(uint64(pageSize)).ToSql()
    if err != nil {
        return nil, err
    }

    var resp []*{{.upperStartCamelObject}}
    err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
    switch err {
    case nil:
        return resp, nil
    default:
        return nil, err
    }
}

// UpdateColsByCond -
func (m *default{{.upperStartCamelObject}}Model) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
