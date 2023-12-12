package nascentcore.ai.modules.system.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import nascentcore.ai.modules.system.domain.Price;
import nascentcore.ai.modules.system.domain.vo.PriceQueryCriteria;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Mapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import java.util.List;

@Mapper
public interface PriceMapper extends BaseMapper<Price> {

    IPage<Price> findAll(@Param("criteria") PriceQueryCriteria criteria, Page<Object> page);

    List<Price> findAll(@Param("criteria") PriceQueryCriteria criteria);
}
