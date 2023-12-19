package nascentcore.ai.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Data
@Configuration
@ConfigurationProperties(prefix = "api")
public class ServiceProperties {
    private Services services = new Services();

    @Data
    public static class Services {
        private String scheduleService;
    }
}
