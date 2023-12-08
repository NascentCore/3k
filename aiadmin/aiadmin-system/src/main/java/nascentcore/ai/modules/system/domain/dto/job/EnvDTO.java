package nascentcore.ai.modules.system.domain.dto.job;

import com.alibaba.fastjson.annotation.JSONField;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class EnvDTO {
    @JSONField(name ="MODIHAND_OPEN_NODE_TOKEN")
    private String modihandOpenNodeToken;
}
