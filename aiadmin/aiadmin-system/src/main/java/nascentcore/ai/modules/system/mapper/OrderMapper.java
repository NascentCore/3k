package nascentcore.ai.modules.system.mapper;


import nascentcore.ai.modules.system.domain.Order;
import nascentcore.ai.modules.system.domain.vo.OrderQueryCriteria;
import java.util.List;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Mapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

@Mapper
public interface OrderMapper extends BaseMapper<Order> {

    IPage<Order> findAll(@Param("criteria") OrderQueryCriteria criteria, Page<Object> page);

    List<Order> findAll(@Param("criteria") OrderQueryCriteria criteria);
}
