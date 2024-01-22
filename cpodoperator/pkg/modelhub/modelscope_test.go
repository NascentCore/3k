package modelhub

import (
	"net/http"
	"reflect"
	"testing"
	"time"
)

var defaultClient = &http.Client{
	Timeout: 5 * time.Second, // 设置超时时间为 5 秒
}

func Test_modelscope_ModelInformation(t *testing.T) {
	type fields struct {
		baseURL    string
		httpClient *http.Client
	}

	type args struct {
		modelID  string
		revision string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    interface{}
		wantErr bool
	}{
		{
			name: "ok",
			fields: fields{
				baseURL:    "https://www.modelscope.cn/",
				httpClient: defaultClient,
			},
			args: args{
				modelID:  "qwen/Qwen-72B-Chat",
				revision: "",
			},
			want:    nil,
			wantErr: false,
		},
		{
			name: "not found",
			fields: fields{
				baseURL:    "https://www.modelscope.cn/",
				httpClient: defaultClient,
			},
			args: args{
				modelID:  "qwen/Qwen-72B-Chat1",
				revision: "",
			},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &modelscope{
				baseURL:    tt.fields.baseURL,
				httpClient: tt.fields.httpClient,
			}
			got, err := m.ModelInformation(tt.args.modelID, tt.args.revision)
			if (err != nil) != tt.wantErr {
				t.Errorf("modelscope.ModelInformation() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("modelscope.ModelInformation() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_modelscope_ModelSize(t *testing.T) {
	type fields struct {
		baseURL    string
		httpClient *http.Client
	}
	type args struct {
		modelID  string
		revision string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    int64
		wantErr bool
	}{
		{
			name: "ok",
			fields: fields{
				baseURL:    "https://www.modelscope.cn/",
				httpClient: defaultClient,
			},
			args: args{
				modelID: "qwen/Qwen-72B-Chat",
			},
			want:    135,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &modelscope{
				baseURL:    tt.fields.baseURL,
				httpClient: tt.fields.httpClient,
			}
			got, err := m.ModelSize(tt.args.modelID, tt.args.revision)
			if (err != nil) != tt.wantErr {
				t.Errorf("modelscope.ModelSize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("modelscope.ModelSize() = %v, want %v", got, tt.want)
			}
		})
	}
}
