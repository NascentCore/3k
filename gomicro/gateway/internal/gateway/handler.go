package gateway

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sxwl/3k/gomicro/gateway/internal/model"

	"github.com/zeromicro/go-zero/core/stores/sqlx"

	"github.com/golang-jwt/jwt/v4"
)

type Handler struct {
	authSecret string
	userModel  model.SysUserModel
}

func NewPassHandler(authSecret, dsn string) *Handler {
	conn := sqlx.NewMysql(dsn)

	return &Handler{
		authSecret: authSecret,
		userModel:  model.NewSysUserModel(conn),
	}
}

func (h *Handler) ServeHTTP(writer http.ResponseWriter, r *http.Request) {
	route, err := GlobalMatcher.Match(r)
	if err != nil {
		http.NotFound(writer, r) // 404
		log.Printf("gateway error=%s", err)
	}

	ok := h.auth(writer, r)
	if ok {
		h.pass(writer, r, route.URL)
	}
}

func (h *Handler) pass(w http.ResponseWriter, r *http.Request, to string) {
	// 解析目标URL
	toUrl, err := url.Parse(to)
	if err != nil {
		http.Error(w, "Error parsing target URL", http.StatusInternalServerError)
		return
	}

	// Read the body
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	bodyReader := bytes.NewReader(bodyBytes)

	// 创建新的请求
	req, err := http.NewRequest(r.Method, toUrl.String(), bodyReader)
	if err != nil {
		http.Error(w, "Error creating request", http.StatusInternalServerError)
		return
	}

	// Set headers
	for key, values := range r.Header {
		req.Header[key] = values
	}
	req.Header.Set("Content-Length", strconv.Itoa(len(bodyBytes)))

	// 创建HTTP客户端并发送请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "Error forwarding request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// 将响应头部复制回原始的响应writer
	for name, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}

	// 设置响应状态码
	w.WriteHeader(resp.StatusCode)

	// 将响应体复制回原始的响应writer
	_, _ = io.Copy(w, resp.Body)
}

func (h *Handler) auth(w http.ResponseWriter, r *http.Request) bool {
	bearerToken := r.Header.Get("Authorization")
	tokenString := strings.TrimPrefix(bearerToken, "Bearer ")
	// Define the key function
	keyFunc := func(token *jwt.Token) (interface{}, error) {
		// Validate the alg is what you expect
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(h.authSecret), nil
	}

	// Parse the token with the specified key function
	token, err := jwt.Parse(tokenString, keyFunc)
	if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
		user, err := h.userModel.FindOneByUsername(context.Background(), sql.NullString{
			String: claims["user"].(string),
			Valid:  true,
		})
		if err != nil {
			http.Error(w, "Unauthorized: "+err.Error(), http.StatusUnauthorized)
			return false
		}

		// Set header
		r.Header.Set("Sx-User", strconv.FormatInt(user.UserId, 10))
		return true
	} else {
		http.Error(w, "Unauthorized: "+err.Error(), http.StatusUnauthorized)
		return false
	}
}
