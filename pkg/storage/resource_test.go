package storage

import (
	"testing"
)

func Test_hash(t *testing.T) {
	type args struct {
		data string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			args: args{data: "models/public/ZhipuAI/chatglm3-6b"},
			want: "10e872cd960e38cb",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hash(tt.args.data); got != tt.want {
				t.Errorf("hash() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExtractTemplate(t *testing.T) {
	type args struct {
		filename string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			args: args{filename: "sxwl-infer-template-gema.md"},
			want: "gema",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ExtractTemplate(tt.args.filename); got != tt.want {
				t.Errorf("ExtractTemplate() = %v, want %v", got, tt.want)
			}
		})
	}
}
