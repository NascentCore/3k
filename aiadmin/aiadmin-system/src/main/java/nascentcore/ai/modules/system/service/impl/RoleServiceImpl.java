package nascentcore.ai.modules.system.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.modules.system.domain.Role;
import nascentcore.ai.modules.system.mapper.RoleMapper;
import nascentcore.ai.modules.system.service.RoleService;
import org.springframework.cache.annotation.CacheConfig;
import org.springframework.stereotype.Service;
import java.util.*;

@Service
@RequiredArgsConstructor
@CacheConfig(cacheNames = "role")
public class RoleServiceImpl extends ServiceImpl<RoleMapper, Role> implements RoleService {

    private final RoleMapper roleMapper;

    @Override
    public List<Role> findByUsersId(Long userId) {
        return roleMapper.findByUserId(userId);
    }
}
