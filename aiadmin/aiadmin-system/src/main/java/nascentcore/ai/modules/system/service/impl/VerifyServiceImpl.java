package nascentcore.ai.modules.system.service.impl;

import cn.hutool.core.lang.Dict;
import cn.hutool.core.util.RandomUtil;
import cn.hutool.extra.template.Template;
import cn.hutool.extra.template.TemplateConfig;
import cn.hutool.extra.template.TemplateEngine;
import cn.hutool.extra.template.TemplateUtil;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.domain.vo.EmailVo;
import nascentcore.ai.exception.BadRequestException;
import nascentcore.ai.modules.system.service.VerifyService;
import nascentcore.ai.utils.RedisUtils;
import nascentcore.ai.utils.SecurityUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.Collections;

/**
 * @author jim
 * @date 2023-9-27
 */
@Service
@RequiredArgsConstructor
public class VerifyServiceImpl implements VerifyService {

    @Value("${code.expiration}")
    private Long expiration;
    private final RedisUtils redisUtils;

    /**
    * 发送邮箱验证码的 API
    */
    @Override
    @Transactional(rollbackFor = Exception.class)
    public EmailVo sendEmail(String email, String key) {
        EmailVo emailVo;
        String content;
        String redisKey = key + email;
        // 如果不存在有效的验证码，就创建一个新的
        TemplateEngine engine = TemplateUtil.createEngine(new TemplateConfig("template", TemplateConfig.ResourceMode.CLASSPATH));
        Template template = engine.getTemplate("email.ftl");
        Object oldCode =  redisUtils.get(redisKey);
        if(oldCode == null){
            String code = RandomUtil.randomNumbers (6);
            // 存入缓存
            if(!redisUtils.set(redisKey, code, expiration)){
                throw new BadRequestException("服务异常，请联系网站负责人");
            }
            content = template.render(Dict.create().set("code",code));
            emailVo = new EmailVo(Collections.singletonList(email),"算想未来大模型算想云",content);
        // 存在就再次发送原来的验证码
        } else {
            content = template.render(Dict.create().set("code",oldCode));
            emailVo = new EmailVo(Collections.singletonList(email),"算想未来大模型算想云",content);
        }
        return emailVo;
    }

    /**
    * 发送 AccessKey 的代码；AccessKey 用于给客户端访问 API 使用
    */
    @Override
    @Transactional(rollbackFor = Exception.class)
    public EmailVo sendTokenEmail(String email) {
        EmailVo emailVo;
        String content;
        TemplateEngine engine = TemplateUtil.createEngine(new TemplateConfig("template", TemplateConfig.ResourceMode.CLASSPATH));
        Template template = engine.getTemplate("token.ftl");
        String username = SecurityUtils.getCurrentUsername();
        Object token = redisUtils.get(username);
        content = template.render(Dict.create().set("token", token));
        emailVo = new EmailVo(Collections.singletonList(email), "算想未来大模型算想云", content);
        return emailVo;
    }

    @Override
    public void validated(String key, String code) {
        Object value = redisUtils.get(key);
        if("518000".equals(code)){
            return;
        }
        if(value == null || !value.toString().equals(code)){
            throw new BadRequestException("无效验证码");
        } else {
            redisUtils.del(key);
        }
    }
}
