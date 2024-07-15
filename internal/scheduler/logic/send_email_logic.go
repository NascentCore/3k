package logic

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sxwl/3k/internal/scheduler/model"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type SendEmailLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewSendEmailLogic(ctx context.Context, svcCtx *svc.ServiceContext) *SendEmailLogic {
	return &SendEmailLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *SendEmailLogic) SendEmail(req *types.SendEmailReq) error {
	VerifyCodeModel := l.svcCtx.VerifyCodeModel

	code, err := VerifyCodeModel.FindOneByQuery(l.ctx, VerifyCodeModel.AllFieldsBuilder().Where(squirrel.Eq{
		"verify_key": req.Email,
	}))
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		l.Errorf("find code email=%s err=%s", req.Email, err)
		return err
	}

	// 1分钟内已经发送过
	if err == nil && time.Since(code.UpdatedAt).Minutes() < 1 {
		return fmt.Errorf("1分钟只能发送一次验证码")
	}

	// 生成6位验证码
	randomCode := fmt.Sprintf("%06d", rand.Intn(1000000))

	if err == nil {
		// 已有该verify_key的记录，更新
		_, err := VerifyCodeModel.UpdateColsByCond(l.ctx, VerifyCodeModel.UpdateBuilder().Where(squirrel.Eq{
			"verify_key": req.Email,
		}).Set("code", randomCode))
		if err != nil {
			l.Errorf("update code email=%s pre=%s new=%s err=%s", req.Email, code.Code, randomCode, err)
			return err
		}
	} else {
		// 创建记录
		_, err = VerifyCodeModel.Insert(l.ctx, &model.VerifyCode{
			VerifyKey: req.Email,
			Code:      randomCode,
		})
		if err != nil {
			l.Errorf("insert code email=%s pre=%s new=%s err=%s", req.Email, code.Code, randomCode, err)
			return err
		}
	}

	// send email
	err = l.svcCtx.EmailSender.SendTemplateEmail([]string{req.Email}, "算想云", "email", map[string]string{"code": randomCode})
	if err != nil {
		return fmt.Errorf("send email err=%s", err)
	}

	return nil
}
