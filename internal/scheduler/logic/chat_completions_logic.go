package logic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

// 上下文 key 类型
type CtxKey string

const (
	CtxKeyResponseWriter CtxKey = "response_writer"
)

type ChatCompletionsLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewChatCompletionsLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ChatCompletionsLogic {
	return &ChatCompletionsLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ChatCompletionsLogic) ChatCompletions(req *types.CompletionReq) (resp *types.CompletionResp, err error) {
	// 将请求体转换为JSON
	if req.Completion.N == 0 {
		req.Completion.N = 1
	}
	if req.Completion.TopP == 0 {
		req.Completion.TopP = 1
	}
	if req.Completion.Temperature == 0 {
		req.Completion.Temperature = 1
	}
	if req.Completion.MaxTokens == 0 {
		req.Completion.MaxTokens = 1000
	}
	reqBody, err := json.Marshal(req.Completion)
	if err != nil {
		return nil, fmt.Errorf("marshal request body failed: %v", err)
	}

	// 创建HTTP请求
	httpReq, err := http.NewRequestWithContext(l.ctx, "POST", l.svcCtx.Config.K8S.PlaygroundUrl, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request failed: %v", err)
	}

	// 设置请求头
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", l.svcCtx.Config.K8S.PlaygroundToken))

	// 发送请求
	client := &http.Client{}
	httpResp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("send request failed: %v", err)
	}
	defer httpResp.Body.Close()

	// 检查响应状态码
	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		return nil, fmt.Errorf("request failed with status code %d: %s", httpResp.StatusCode, string(respBody))
	}

	// 如果是流式响应，直接转发响应体
	if req.Completion.Stream {
		w, ok := l.ctx.Value(CtxKeyResponseWriter).(http.ResponseWriter)
		if !ok {
			return nil, fmt.Errorf("failed to get response writer from context")
		}

		// 使用缓冲读取器来读取响应
		reader := bufio.NewReader(httpResp.Body)
		for {
			// 读取一行数据
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("read stream failed: %v", err)
			}

			// 写入到客户端
			if _, err := w.Write(line); err != nil {
				return nil, fmt.Errorf("write stream failed: %v", err)
			}
			w.(http.Flusher).Flush()
		}
		return nil, nil
	}

	// 非流式响应，读取完整响应体
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body failed: %v", err)
	}

	// 解析响应体
	resp = &types.CompletionResp{}
	if err := json.Unmarshal(respBody, resp); err != nil {
		return nil, fmt.Errorf("unmarshal response body failed: %v", err)
	}

	return resp, nil
}
