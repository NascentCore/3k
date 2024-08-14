package logic

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/bcrypt"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/rsa"
	strings2 "sxwl/3k/pkg/strings"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/golang-jwt/jwt/v4"
	"github.com/google/uuid"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type UserLoginLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewUserLoginLogic(ctx context.Context, svcCtx *svc.ServiceContext) *UserLoginLogic {
	return &UserLoginLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *UserLoginLogic) UserLogin(req *types.LoginReq) (resp *types.LoginResp, err error) {
	UserModel := l.svcCtx.UserModel
	VerifyCodeModel := l.svcCtx.VerifyCodeModel

	user, err := UserModel.FindOneByUsername(l.ctx, orm.NullString(req.Username))
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		l.Errorf("UserLogin username=%s err=%s", req.Username, err)
		return nil, ErrSystem
	}

	if errors.Is(err, model.ErrNotFound) {
		// 走注册流程

		// 生成随机密码
		randomPassword, err := strings2.RandomString(10, true, true, true)
		if err != nil {
			return nil, err
		}

		// RSA加密
		passwordRSAEncrypted, err := rsa.Encrypt(randomPassword, l.svcCtx.Config.Rsa.PublicKey)
		if err != nil {
			return nil, err
		}

		err = NewRegisterLogic(l.ctx, l.svcCtx).Register(&types.RegisterUserReq{
			Code:     req.Code,
			Username: req.Username,
			Email:    req.Username,
			Enabled:  1,
			Password: passwordRSAEncrypted,
			From:     "login",
		})
		if err != nil {
			return nil, err
		}

		// 重新获取user
		user, err = UserModel.FindOneByUsername(l.ctx, orm.NullString(req.Username))
		if err != nil {
			l.Errorf("UserLogin fetch user username=%s err=%s", req.Username, err)
			return nil, ErrSystem
		}
	}

	// check password or email code
	if req.Password != "" {
		password, err := rsa.Decrypt(req.Password, l.svcCtx.Config.Rsa.PrivateKey)
		if err != nil {
			l.Errorf("UserLogin decrypt username=%s err=%s", req.Username, err)
			return nil, ErrSystem
		}

		if !bcrypt.CheckPasswordHash(password, user.Password.String) {
			return nil, ErrPassword
		}
	} else if req.Code != "" {
		// check verify code is correct
		code, err := VerifyCodeModel.FindOneByVerifyKey(l.ctx, req.Username)
		if err != nil {
			return nil, ErrCode
		}

		if time.Since(code.UpdatedAt).Minutes() > 5 {
			return nil, ErrCodeTimeout
		}

		if code.Code != req.Code {
			return nil, ErrCode
		}
	} else {
		return nil, ErrPassword
	}

	resp = &types.LoginResp{User: types.WrapUser{User: types.UserInfo{}}}
	_ = copier.Copy(&resp.User.User, user)

	// createTime
	resp.User.User.CreateTime = user.CreateTime.Time.Format(time.DateTime)
	// updateTime
	resp.User.User.UpdateTime = user.UpdateTime.Time.Format(time.DateTime)
	// enabled
	if user.Enabled.Valid && user.Enabled.Int64 > 0 {
		resp.User.User.Enabled = true
	}
	// id
	resp.User.User.ID = user.UserId
	// user_id
	resp.User.User.UserID = user.NewUserId
	// isAdmin
	if user.Admin > 0 {
		resp.User.User.IsAdmin = true
	}
	// TODO remove password

	// jwt token
	// Define the secret key. This should be a securely stored secret in production.
	secret := []byte(l.svcCtx.Config.Auth.Secret)

	// create jti token的唯一id
	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userName: %s err: %s", req.Username, err)
		return nil, err
	}
	formattedUUID := strings.Replace(newUUID.String(), "-", "", -1)

	// TODO 目前是一个永不过期的token，不太合适
	claims := jwt.MapClaims{
		"jti":      formattedUUID,
		"username": user.Username.String,
		"userid":   user.UserId,
		"user_id":  user.NewUserId,
		"sub":      user.Username.String,
	}

	// Create token with signing method HS512
	token := jwt.NewWithClaims(jwt.SigningMethodHS512, claims)

	// Sign the token with the secret key
	signedToken, err := token.SignedString(secret)
	if err != nil {
		l.Errorf("token sign userName: %s err: %s", req.Username, err)
		return nil, err
	}

	resp.Token = fmt.Sprintf("Bearer %s", signedToken)
	return
}
