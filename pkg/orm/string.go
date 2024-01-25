package orm

import (
	"database/sql"
	"time"
)

func NullString(s string) sql.NullString {
	return sql.NullString{String: s, Valid: true}
}

func NullTime(t time.Time) sql.NullTime {
	return sql.NullTime{Time: t, Valid: true}
}

func NullInt64(i int64) sql.NullInt64 {
	return sql.NullInt64{Int64: i, Valid: true}
}
