package logic

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/golang-jwt/jwt/v4"
	"github.com/google/uuid"
	"github.com/zeromicro/go-zero/core/logx"
)

type DingtalkUserinfoLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewDingtalkUserinfoLogic(ctx context.Context, svcCtx *svc.ServiceContext) *DingtalkUserinfoLogic {
	return &DingtalkUserinfoLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *DingtalkUserinfoLogic) DingtalkUserinfo(req *types.DingCallbackReq) (resp *types.LoginResp, err error) {
	// 使用配置中的AppKey和AppSecret
	appKey := l.svcCtx.Config.DingTalk.AppKey
	appSecret := l.svcCtx.Config.DingTalk.AppSecret

	// 生成 timestamp 和 signature
	timestamp := fmt.Sprintf("%d", time.Now().UnixNano()/int64(time.Millisecond))
	signature := generateSignature(timestamp, appSecret)

	// 使用 access_token 获取用户信息
	userInfoURL := fmt.Sprintf("https://oapi.dingtalk.com/sns/getuserinfo_bycode?accessKey=%s&timestamp=%s&signature=%s", appKey, timestamp, signature)
	respUserInfo, err := http.Post(userInfoURL, "application/json", strings.NewReader(fmt.Sprintf(`{"tmp_auth_code":"%s"}`, req.Code)))
	if err != nil {
		return nil, err
	}
	defer respUserInfo.Body.Close()

	// 解析用户信息
	var userInfoResponse struct {
		UserInfo struct {
			Nick    string `json:"nick"`
			UnionId string `json:"unionid"`
			OpenId  string `json:"openid"`
		} `json:"user_info"`
		ErrCode int    `json:"errcode"`
		ErrMsg  string `json:"errmsg"`
	}
	if err := json.NewDecoder(respUserInfo.Body).Decode(&userInfoResponse); err != nil {
		return nil, err
	}
	if userInfoResponse.ErrCode != 0 {
		return nil, fmt.Errorf(userInfoResponse.ErrMsg)
	}

	resp = &types.LoginResp{User: types.WrapUser{User: types.UserInfo{}}}
	resp.User.User.UserID = userInfoResponse.UserInfo.OpenId
	resp.User.User.Username = userInfoResponse.UserInfo.Nick
	resp.User.User.IsAdmin = false

	// jwt token
	// Define the secret key. This should be a securely stored secret in production.
	secret := []byte(l.svcCtx.Config.Auth.Secret)

	// create jti token的唯一id
	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userName: %s err: %s", userInfoResponse.UserInfo.Nick, err)
		return nil, err
	}
	formattedUUID := strings.Replace(newUUID.String(), "-", "", -1)

	// TODO 目前是一个永不过期的token，不太合适
	claims := jwt.MapClaims{
		"jti":      formattedUUID,
		"username": userInfoResponse.UserInfo.Nick,
		"userid":   0,
		"user_id":  userInfoResponse.UserInfo.OpenId,
		"sub":      userInfoResponse.UserInfo.Nick,
	}

	// Create token with signing method HS512
	token := jwt.NewWithClaims(jwt.SigningMethodHS512, claims)

	// Sign the token with the secret key
	signedToken, err := token.SignedString(secret)
	if err != nil {
		l.Errorf("token sign userName: %s err: %s", userInfoResponse.UserInfo.Nick, err)
		return nil, err
	}

	resp.Token = fmt.Sprintf("Bearer %s", signedToken)
	return
}

func generateSignature(timestamp, appSecret string) string {
	// 使用 timestamp 作为 stringToSign
	stringToSign := timestamp

	// 使用 HmacSHA256 加密
	h := hmac.New(sha256.New, []byte(appSecret))
	h.Write([]byte(stringToSign))
	signature := base64.StdEncoding.EncodeToString(h.Sum(nil))

	// 对签名结果进行URL编码，并处理特殊字符
	encodedSignature := url.QueryEscape(signature)
	urlEncodedSignature := strings.Replace(encodedSignature, "+", "%20", -1)
	urlEncodedSignature = strings.Replace(urlEncodedSignature, "*", "%2A", -1)
	urlEncodedSignature = strings.Replace(urlEncodedSignature, "~", "%7E", -1)
	urlEncodedSignature = strings.Replace(urlEncodedSignature, "/", "%2F", -1)

	return urlEncodedSignature
}
