package nascentcore.ai.modules.system.service.impl;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import nascentcore.ai.modules.system.domain.CpodMain;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import nascentcore.ai.modules.system.service.CpodMainService;
import nascentcore.ai.modules.system.domain.vo.CpodMainQueryCriteria;
import nascentcore.ai.modules.system.mapper.CpodMainMapper;
import nascentcore.ai.utils.PageResult;
import nascentcore.ai.utils.PageUtil;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.List;

/**
* @description 服务实现
* @author jimi
* @date 2023-10-23
**/
@Service
@RequiredArgsConstructor
public class CpodMainServiceImpl extends ServiceImpl<CpodMainMapper, CpodMain> implements CpodMainService {

    private final CpodMainMapper cpodMainMapper;
    @Override
    public PageResult<CpodMain> queryAll(CpodMainQueryCriteria criteria, Page<Object> page){
        return PageUtil.toPage(cpodMainMapper.findAll(criteria, page));
    }
    @Override
    public List<CpodMain> queryAll(CpodMainQueryCriteria criteria){
        return cpodMainMapper.findAll(criteria);
    }

    @Override
    public List<CpodMain> queryAllGpuType(){
        return cpodMainMapper.queryAllGpuType();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void create(CpodMain resources) {
        save(resources);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(CpodMain resources) {
        CpodMain cpodMain = getById(resources.getMainId());
        cpodMain.copy(resources);
        saveOrUpdate(cpodMain);
    }

    @Override
    public List<CpodMain> findByCpodId(String cpodid) {
        return cpodMainMapper.findByCpodId(cpodid);
    }
}
