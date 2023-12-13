package nascentcore.ai.modules.system.mapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import nascentcore.ai.modules.system.domain.Config;
import nascentcore.ai.modules.system.domain.vo.ConfigQueryCriteria;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface ConfigMapper extends BaseMapper<Config> {

    List<Config> findAll(@Param("criteria") ConfigQueryCriteria criteria);
}
