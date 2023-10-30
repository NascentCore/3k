package nascentcore.ai.modules.system.mapper;

import nascentcore.ai.modules.system.domain.CpodMain;
import nascentcore.ai.modules.system.domain.vo.CpodMainQueryCriteria;
import java.util.List;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Mapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

/**
* @author jimi
* @date 2023-10-23
**/
@Mapper
public interface CpodMainMapper extends BaseMapper<CpodMain> {

    IPage<CpodMain> findAll(@Param("criteria") CpodMainQueryCriteria criteria, Page<Object> page);

    List<CpodMain> findAll(@Param("criteria") CpodMainQueryCriteria criteria);

    List<CpodMain> findByCpodId(@Param("cpodid") String cpodid);

    List<CpodMain> queryAllGpuType();
}
