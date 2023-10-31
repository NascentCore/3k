package nascentcore.ai.modules.system.service;

import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.modules.system.domain.User;

/**
 * @author jim
 * @date 2023-9-27
 */
public interface UserService extends IService<User> {
    /**
     * 新增用户
     * @param resources /
     */
    void create(User resources);
    /**
     * 根据用户名查询
     * @param userName /
     * @return /
     */
    User getLoginData(String userName);
}
