package logic

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/bcrypt"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/rsa"
	uuid2 "sxwl/3k/pkg/uuid"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/golang-jwt/jwt/v4"
	"github.com/google/uuid"
	"github.com/zeromicro/go-zero/core/logx"
)

type RegisterLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewRegisterLogic(ctx context.Context, svcCtx *svc.ServiceContext) *RegisterLogic {
	return &RegisterLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *RegisterLogic) Register(req *types.RegisterUserReq) error {
	UserModel := l.svcCtx.UserModel
	VerifyCodeModel := l.svcCtx.VerifyCodeModel
	UserBalanceModel := l.svcCtx.UserBalanceModel
	RechargeModel := l.svcCtx.RechargeModel

	// check verify code is correct
	code, err := VerifyCodeModel.FindOneByVerifyKey(l.ctx, req.Email)
	if err != nil {
		return fmt.Errorf("验证码错误")
	}

	if time.Since(code.UpdatedAt).Minutes() > 5 {
		return fmt.Errorf("验证码已过期，请重新获取。")
	}

	if code.Code != req.Code {
		return fmt.Errorf("输入的验证码有误，请检查后重试。")
	}

	// check username and email not exists
	_, err = UserModel.FindOneByUsername(l.ctx, orm.NullString(req.Username))
	if !errors.Is(err, model.ErrNotFound) {
		return fmt.Errorf("此用户名已经注册，请尝试使用其他用户名。")
	}

	_, err = UserModel.FindOneByEmail(l.ctx, orm.NullString(req.Email))
	if !errors.Is(err, model.ErrNotFound) {
		return fmt.Errorf("此邮箱已被注册，请尝试其他邮箱。")
	}

	// password encrypt
	password, err := rsa.Decrypt(req.Password, l.svcCtx.Config.Rsa.PrivateKey)
	if err != nil {
		l.Errorf("register decrypt username=%s err=%s", req.Username, err)
		return fmt.Errorf("系统错误，请联系管理员")
	}

	passwordHash, err := bcrypt.GeneratePasswordHash(password)
	if err != nil {
		l.Errorf("register hash username=%s err=%s", req.Username, err)
		return fmt.Errorf("系统错误，请联系管理员")
	}

	// create user_id
	userID, err := uuid2.WithPrefix("user")
	if err != nil {
		l.Errorf("register new user_id err=%s", err)
		return ErrSystem
	}

	// create user
	user := &model.SysUser{
		NewUserId:  userID,
		Username:   orm.NullString(req.Username),
		Email:      orm.NullString(req.Email),
		Password:   orm.NullString(passwordHash),
		Enabled:    orm.NullInt64(int64(req.Enabled)),
		CreateBy:   orm.NullString("System"),
		UpdateBy:   orm.NullString("System"),
		CreateTime: orm.NullTime(time.Now()),
		UpdateTime: orm.NullTime(time.Now()),
	}

	if req.UserType == model.SysUserSupplier {
		user.UserType = model.SysUserSupplier
		user.CompanyName = orm.NullString(req.CompanyName)
		user.CompanyPhone = orm.NullString(req.CompanyPhone)

		newUUID, err := uuid.NewRandom()
		if err != nil {
			l.Errorf("newUUID username=%s err=%s", req.Username, err)
			return fmt.Errorf("注册失败 err=%s", err)
		}

		user.CompanyId = orm.NullString(newUUID.String())
	} else {
		user.UserType = model.SysUserConsumer
	}
	_, err = UserModel.Insert(l.ctx, user)
	if err != nil {
		l.Errorf("insert user username=%s err=%s", req.Username, err)
		return fmt.Errorf("注册失败 err=%s", err)
	}

	// 余额初始化
	user, err = UserModel.FindOneByUsername(l.ctx, sql.NullString{
		String: req.Username,
		Valid:  true,
	})
	if err != nil {
		l.Errorf("find user by username=%s err=%s", req.Username, err)
		return fmt.Errorf("注册失败 err=%s", err)
	}

	balance := &model.UserBalance{
		UserId:    user.UserId, // TODO remove
		NewUserId: user.NewUserId,
		Balance:   l.svcCtx.Config.Billing.InitBalance,
	}
	_, err = UserBalanceModel.Insert(l.ctx, balance)
	if err != nil {
		l.Errorf("insert balance username=%s err=%s", req.Username, err)
		return fmt.Errorf("注册失败 err=%s", err)
	}

	recharge, err := uuid2.WithPrefix("recharge")
	if err != nil {
		l.Errorf("new uuid username=%s err=%s", req.Username, err)
		return fmt.Errorf("注册失败 err=%s", err)
	}
	_, err = RechargeModel.Insert(l.ctx, &model.UserRecharge{
		RechargeId:    recharge,
		UserId:        userID,
		Amount:        l.svcCtx.Config.Billing.InitBalance,
		BeforeBalance: 0,
		AfterBalance:  l.svcCtx.Config.Billing.InitBalance,
		Description:   DescRechargeRegister,
	})
	if err != nil {
		l.Errorf("insert recharge username=%s err=%s", req.Username, err)
		return fmt.Errorf("注册失败 err=%s", err)
	}

	// send token email
	// jwt token
	secret := []byte(l.svcCtx.Config.Auth.Secret)

	// create jti token的唯一id
	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userName: %s err: %s", req.Username, err)
		return err
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
		return err
	}

	if req.From == "" {
		err = l.svcCtx.EmailSender.SendTemplateEmail([]string{req.Email}, "算想云", "token", map[string]string{"token": signedToken})
		if err != nil {
			return fmt.Errorf("send email err=%s", err)
		}
	} else {
		err = l.svcCtx.EmailSender.SendTemplateEmail([]string{req.Email}, "算想云", "token_pass", map[string]string{
			"token":    signedToken,
			"password": password,
		})
		if err != nil {
			return fmt.Errorf("send email err=%s", err)
		}
	}

	return nil
}
