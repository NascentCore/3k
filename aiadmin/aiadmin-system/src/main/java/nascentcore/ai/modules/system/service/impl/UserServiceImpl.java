package nascentcore.ai.modules.system.service.impl;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.exception.EntityNotFoundException;
import nascentcore.ai.modules.system.domain.User;
import nascentcore.ai.exception.EntityExistException;
import nascentcore.ai.modules.system.domain.vo.UserQueryCriteria;
import nascentcore.ai.modules.system.mapper.UserMapper;
import nascentcore.ai.modules.system.service.UserService;
import nascentcore.ai.utils.PageResult;
import nascentcore.ai.utils.PageUtil;
import nascentcore.ai.utils.RedisUtils;
import org.springframework.cache.annotation.CacheConfig;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * @author jim
 * @date 2023-9-27
 */
@Service
@RequiredArgsConstructor
@CacheConfig(cacheNames = "user")
public class UserServiceImpl extends ServiceImpl<UserMapper, User> implements UserService {

    private final UserMapper userMapper;
    private final RedisUtils redisUtils;
    @Override
    @Transactional(rollbackFor = Exception.class)
    public PageResult<User> queryAll(UserQueryCriteria criteria, Page<Object> page) {
        List<User> users = userMapper.findAll(criteria);
        if(null != users && !users.isEmpty()){
            users.get(0).setAccessKey((String)redisUtils.get(users.get(0).getUsername()));
        }
        Long total = userMapper.countAll(criteria);
        return PageUtil.toPage(users, total);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void create(User resources) {
        if (userMapper.findByUsername(resources.getUsername()) != null) {
            throw new EntityExistException(User.class, "username", resources.getUsername());
        }
        if (userMapper.findByEmail(resources.getEmail()) != null) {
            throw new EntityExistException(User.class, "email", resources.getEmail());
        }
        if (userMapper.findByPhone(resources.getPhone()) != null) {
            throw new EntityExistException(User.class, "phone", resources.getPhone());
        }
        save(resources);
    }
    @Override
    @Transactional(rollbackFor = Exception.class)
    public User getLoginData(String userName) {
        User user = userMapper.findByUsername(userName);
        if (user == null) {
            throw new EntityNotFoundException(User.class, "name", userName);
        } else {
            return user;
        }
    }
}
