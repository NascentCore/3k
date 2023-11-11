package nascentcore.ai.modules.system.service.impl;

import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.mapper.UserJobMapper;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import nascentcore.ai.modules.system.service.UserJobService;
import nascentcore.ai.modules.system.domain.vo.UserJobQueryCriteria;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import nascentcore.ai.utils.PageUtil;
import nascentcore.ai.utils.PageResult;

import java.util.List;
import java.util.Set;

/**
 * @description 服务实现
 * @author jim
 * @date 2023-10-12
 **/
@Service
@RequiredArgsConstructor
public class UserJobServiceImpl extends ServiceImpl<UserJobMapper, UserJob> implements UserJobService {

    private final UserJobMapper userJobMapper;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PageResult<UserJob> queryAll(UserJobQueryCriteria criteria, Page<Object> page){
        return PageUtil.toPage(userJobMapper.findAll(criteria, page));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void create(UserJob resources) {
        save(resources);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<UserJob> queryAll(UserJobQueryCriteria criteria){
        return userJobMapper.findAll(criteria);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void save(List<UserJob> columnInfos) {
        saveOrUpdateBatch(columnInfos);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void delete(Set<Long> ids) {
        for (Long id : ids) {
            removeById(id);
        }
    }
}
