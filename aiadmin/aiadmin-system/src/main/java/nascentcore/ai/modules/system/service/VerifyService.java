package nascentcore.ai.modules.system.service;

import nascentcore.ai.domain.vo.EmailVo;

/**
 * @author jim
 * @date 2023-9-27
 */
public interface VerifyService {

    /**
     * 发送验证码
     * @param email /
     * @param key /
     * @return /
     */
    EmailVo sendEmail(String email, String key);


    /**
     * 验证
     * @param code /
     * @param key /
     */
    void validated(String key, String code);
}
