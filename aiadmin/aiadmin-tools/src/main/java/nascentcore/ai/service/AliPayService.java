package nascentcore.ai.service;

import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.domain.vo.TradeVo;
import nascentcore.ai.domain.AlipayConfig;

public interface AliPayService extends IService<AlipayConfig> {

    /**
     * 查询配置
     *
     * @return AlipayConfig
     */
    AlipayConfig find();

    /**
     * 更新配置
     *
     * @param alipayConfig 支付宝配置
     * @return AlipayConfig
     */
    AlipayConfig config(AlipayConfig alipayConfig) throws Exception;

    /**
     * 处理来自当面付的交易请求
     *
     * @param alipay 支付宝配置
     * @param trade  交易详情
     * @return String
     * @throws Exception 异常
     */
    String toPayAsPrecreate(AlipayConfig alipay, TradeVo trade);
}
