package nascentcore.ai.modules.system.service.impl;

import nascentcore.ai.modules.system.domain.Fileurl;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import nascentcore.ai.modules.system.service.FileurlService;
import nascentcore.ai.modules.system.domain.vo.FileurlQueryCriteria;
import nascentcore.ai.modules.system.mapper.FileurlMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.List;

@Service
@RequiredArgsConstructor
public class FileurlServiceImpl extends ServiceImpl<FileurlMapper, Fileurl> implements FileurlService {
    private final FileurlMapper fileurlMapper;
    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<Fileurl> queryAll(FileurlQueryCriteria criteria){
        return fileurlMapper.findAll(criteria);
    }
    @Override
    @Transactional(rollbackFor = Exception.class)
    public void save(List<Fileurl> columnInfos) {
        saveOrUpdateBatch(columnInfos);
    }
}
