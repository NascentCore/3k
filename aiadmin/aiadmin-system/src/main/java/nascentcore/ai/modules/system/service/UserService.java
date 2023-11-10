package nascentcore.ai.modules.system.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.modules.system.domain.User;
import nascentcore.ai.modules.system.domain.vo.UserQueryCriteria;
import nascentcore.ai.utils.PageResult;

/**
 * @author jim
 * @date 2023-9-27
 */
public interface UserService extends IService<User> {
    /**
     * 查询全部
     *
     * @param criteria 条件
     * @param page     分页参数
     * @return /
     */
    PageResult<User> queryAll(UserQueryCriteria criteria, Page<Object> page);
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
