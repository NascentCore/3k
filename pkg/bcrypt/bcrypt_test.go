package bcrypt

import (
	"testing"
)

func TestCheckPasswordHash(t *testing.T) {
	type args struct {
		password string
		hash     string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			args: args{
				password: "123456",
				hash:     "$2a$10$DkpjKA6lopLZ2kE7yh65Q.BygUPddvhu.4WMaWaEINABYPhuhvRUG",
			},
			want: true,
		},
		{
			args: args{
				password: "123456",
				hash:     "$2a$10$uLQJ4/y59oB7JVHmJkzPAOfUZ5xSuqMD8xbnfanEkrkSLhYEqoUAy",
			},
			want: true,
		},
		{
			args: args{
				password: "123456",
				hash:     "$2a$10$uLQJ4/y59oB7JVHmJkzPAOfUZ5xSuqMD8xbnfanEkrkSLhYEqbUAy",
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CheckPasswordHash(tt.args.password, tt.args.hash); got != tt.want {
				t.Errorf("CheckPasswordHash() = %v, want %v", got, tt.want)
			}
		})
	}
}
