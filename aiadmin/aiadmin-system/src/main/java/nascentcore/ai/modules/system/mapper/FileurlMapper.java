package nascentcore.ai.modules.system.mapper;

import nascentcore.ai.modules.system.domain.Fileurl;
import nascentcore.ai.modules.system.domain.vo.FileurlQueryCriteria;
import java.util.List;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Mapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;

@Mapper
public interface FileurlMapper extends BaseMapper<Fileurl> {
    List<Fileurl> findAll(@Param("criteria") FileurlQueryCriteria criteria);
}
