package nascentcore.ai.service.impl;

import com.alipay.api.AlipayApiException;
import com.alipay.api.AlipayClient;
import com.alipay.api.DefaultAlipayClient;
import com.alipay.api.domain.AlipayTradePrecreateModel;
import com.alipay.api.request.AlipayTradePagePayRequest;
import com.alipay.api.request.AlipayTradePrecreateRequest;
import com.alipay.api.response.AlipayTradePrecreateResponse;

// Batis ORM
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;

import lombok.RequiredArgsConstructor;

import org.springframework.cache.annotation.CacheConfig;
import org.springframework.cache.annotation.CachePut;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import nascentcore.ai.domain.vo.TradeVo;
import nascentcore.ai.domain.AlipayConfig;
import nascentcore.ai.exception.BadRequestException;
import nascentcore.ai.mapper.AliPayConfigMapper;
import nascentcore.ai.service.AliPayService;
import nascentcore.ai.utils.EncryptUtils;

@Service
@RequiredArgsConstructor
@CacheConfig(cacheNames = "aliPay")
public class AliPayServiceImpl extends ServiceImpl<AliPayConfigMapper, AlipayConfig> implements AliPayService {

    @Override
    public AlipayConfig find() {
        AlipayConfig alipayConfig = getById(1L);
        return alipayConfig == null ? new AlipayConfig() : alipayConfig;
    }

    @Override
    @CachePut(key = "'config'")
    @Transactional(rollbackFor = Exception.class)
    public AlipayConfig config(AlipayConfig alipayConfig) throws Exception {
        alipayConfig.setId(1L);
        alipayConfig.setPrivateKey(EncryptUtils.desEncrypt(alipayConfig.getPrivateKey()));
        alipayConfig.setPublicKey(EncryptUtils.desEncrypt(alipayConfig.getPublicKey()));
        saveOrUpdate(alipayConfig);
        return alipayConfig;
    }

    @Override
    public String toPayAsPrecreate(AlipayConfig alipay, TradeVo trade) {
        if (alipay.getId() == null) {
            throw new BadRequestException("请先添加相应配置，再操作");
        }
        try {
            // 对称解密
            alipay.setPrivateKey(EncryptUtils.desDecrypt(alipay.getPrivateKey()));
            alipay.setPublicKey(EncryptUtils.desDecrypt(alipay.getPublicKey()));
        } catch (Exception e) {
            throw new BadRequestException(e.getMessage());
        }
        AlipayClient alipayClient = new DefaultAlipayClient(alipay.getGatewayUrl(), alipay.getAppId(), alipay.getPrivateKey(), alipay.getFormat(), alipay.getCharset(), alipay.getPublicKey(), alipay.getSignType());

        double money = Double.parseDouble(trade.getTotalAmount());
        double maxMoney = 5000;
        if (money <= 0 || money >= maxMoney) {
            throw new BadRequestException("支付金额过大");
        }
        // 创建API对应的request
        AlipayTradePrecreateRequest request = new AlipayTradePrecreateRequest();
        AlipayTradePrecreateModel model = new AlipayTradePrecreateModel();
        model.setOutTradeNo(trade.getOutTradeNo());
        model.setTotalAmount(trade.getTotalAmount());
        model.setSubject(trade.getSubject());
        request.setBizModel(model);
        request.setNotifyUrl(alipay.getNotifyUrl());
        //3.发起请求获取响应
        try {
            AlipayTradePrecreateResponse response = alipayClient.execute(request);
            System.out.println(response.getBody());
            //4.验证是否成功
            if (response.isSuccess()) {
                //5.返回支付的二维码的串码
                return response.getQrCode();
            }
        } catch (AlipayApiException e) {
            System.out.println(e.getErrMsg());
        }
        return null;
    }
}
