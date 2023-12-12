package nascentcore.ai.modules.system.service.impl;


import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import nascentcore.ai.modules.system.domain.Price;
import nascentcore.ai.modules.system.domain.vo.PriceQueryCriteria;
import nascentcore.ai.modules.system.mapper.PriceMapper;
import nascentcore.ai.modules.system.service.PriceService;
import nascentcore.ai.utils.PageResult;
import nascentcore.ai.utils.PageUtil;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.List;

@Service
@RequiredArgsConstructor
public class PriceServiceImpl extends ServiceImpl<PriceMapper, Price> implements PriceService {

    private final PriceMapper priceMapper;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PageResult<Price> queryAll(PriceQueryCriteria criteria, Page<Object> page){
        return PageUtil.toPage(priceMapper.findAll(criteria, page));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<Price> queryAll(PriceQueryCriteria criteria){
        return priceMapper.findAll(criteria);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Price queryByprod(String gpuprod){
        PriceQueryCriteria criteria = new PriceQueryCriteria();
        criteria.setGpuProd(gpuprod);
        List<Price> priceList = priceMapper.findAll(criteria);
        if(null != priceList && !priceList.isEmpty()){
            return priceList.get(0);
        }else {
            return null;
        }
    }
    @Override
    @Transactional(rollbackFor = Exception.class)
    public void create(Price resources) {
        save(resources);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(Price resources) {
        Price price = getById(resources.getPriceId());
        price.copy(resources);
        saveOrUpdate(price);
    }
}
