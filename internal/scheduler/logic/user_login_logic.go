package logic

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/orm"
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

	user, err := UserModel.FindOneByUsername(l.ctx, orm.NullString(req.Username))
	if errors.Is(err, model.ErrNotFound) {
		err = fmt.Errorf("用户名或密码不正确")
		return nil, err
	}

	if err != nil {
		l.Errorf("UserLogin username=%s err=%s", req.Username, err)
		return nil, err
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
	resp.User.User.ID = int(user.UserId)
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