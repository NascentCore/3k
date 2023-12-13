package nascentcore.ai.modules.system.service.impl;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.domain.AlipayConfig;
import nascentcore.ai.domain.vo.TradeVo;
import nascentcore.ai.modules.system.domain.AliPayRes;
import nascentcore.ai.modules.system.domain.Order;
import nascentcore.ai.modules.system.domain.Price;
import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.domain.vo.UserJobQueryCriteria;
import nascentcore.ai.modules.system.mapper.OrderMapper;
import nascentcore.ai.modules.system.service.OrderService;
import nascentcore.ai.modules.system.domain.vo.OrderQueryCriteria;
import nascentcore.ai.modules.system.service.PriceService;
import nascentcore.ai.modules.system.service.UserJobService;
import nascentcore.ai.service.AliPayService;
import nascentcore.ai.utils.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.List;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
public class OrderServiceImpl extends ServiceImpl<OrderMapper, Order> implements OrderService {
    private final OrderMapper orderMapper;
    private final AliPayService alipayService;
    private final AlipayUtils alipayUtils;
    private final UserJobService userJobService;
    private final PriceService priceService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PageResult<Order> queryAll(OrderQueryCriteria criteria, Page<Object> page) {
        return PageUtil.toPage(orderMapper.findAll(criteria, page));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<Order> queryAll(OrderQueryCriteria criteria) {
        return orderMapper.findAll(criteria);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Order queryOne(OrderQueryCriteria criteria) {
        List<Order> orderList = orderMapper.findAll(criteria);
        if (null != orderList && !orderList.isEmpty()) {
            return orderList.get(0);
        } else {
            return null;
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Order querybyNo(String outTradeNo) {
        OrderQueryCriteria criteria = new OrderQueryCriteria();
        criteria.setOutTradeNo(outTradeNo);
        List<Order> orderList = orderMapper.findAll(criteria);
        if (null != orderList && !orderList.isEmpty()) {
            return orderList.get(0);
        } else {
            return null;
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void create(Order resources) {
        save(resources);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(Order resources) {
        Order order = getById(resources.getOrderId());
        order.copy(resources);
        saveOrUpdate(order);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public AliPayRes getPayInfo(String jobName) {
        OrderQueryCriteria orderQueryCriteria = new OrderQueryCriteria();
        orderQueryCriteria.setJobName(jobName);
        Order order = queryOne(orderQueryCriteria);
        AliPayRes aliPayRes = new AliPayRes();
        if (null != order) {
            if (order.getStatus() == AlipayUtils.STATUS_PAY) {
                //已付款返回显示下载模型
                aliPayRes.setState(String.valueOf(AlipayUtils.STATUS_PAY));
            } else {
                //未付款取订单号，继续使用原订单号支付
                aliPayRes.setState(String.valueOf(AlipayUtils.STATUS_NOTPAY));
                aliPayRes.setSubject(order.getSubject());
                aliPayRes.setTotalAmount(String.valueOf(order.getTotalAmount()));
                aliPayRes.setBody(order.getBody());
                String payUrl = getPayUrl(order.getOutTradeNo(), aliPayRes.getSubject(), aliPayRes.getTotalAmount());
                aliPayRes.setQrCode(payUrl);
            }
        } else {
            //获取额度，创建支付订单，返回新订单信息
            TradeVo trade = getPriceInfo(jobName);
            if (Double.parseDouble(trade.getTotalAmount()) < 2) {
                aliPayRes.setState(String.valueOf(AlipayUtils.STATUS_PAY));
            } else {
                //插入支付订单
                Order orders = new Order();
                orders.setSubject(trade.getSubject());
                orders.setTotalAmount(Double.valueOf(trade.getTotalAmount()));
                orders.setBody(trade.getBody());
                orders.setStatus(AlipayUtils.STATUS_NOTPAY);
                orders.setUserId(SecurityUtils.getCurrentUserId());
                orders.setOutTradeNo(trade.getOutTradeNo());
                orders.setCreateTime(DateUtil.getUTCTimeStamp());
                orders.setUpdateTime(DateUtil.getUTCTimeStamp());
                orders.setJobName(jobName);
                create(orders);
                aliPayRes.setState(String.valueOf(AlipayUtils.STATUS_NOTPAY));
                String payUrl = getPayUrl(trade.getOutTradeNo(), trade.getSubject(), trade.getTotalAmount());
                aliPayRes.setQrCode(payUrl);
                aliPayRes.setSubject(trade.getSubject());
                aliPayRes.setTotalAmount(String.valueOf(trade.getTotalAmount()));
                aliPayRes.setBody(trade.getBody());
            }
        }
        return aliPayRes;
    }

    private String getPayUrl(String outTradeNo, String subject, String totalAmount) {
        TradeVo trade = new TradeVo();
        trade.setOutTradeNo(outTradeNo);
        trade.setSubject(subject);
        trade.setTotalAmount(totalAmount);
        AlipayConfig alipay = alipayService.find();
        return alipayService.toPayAsPrecreate(alipay, trade);
    }

    private TradeVo getPriceInfo(String jobname) {
        TradeVo trade = new TradeVo();
        trade.setOutTradeNo(alipayUtils.getOrderCode());
        trade.setSubject("算力服务消费");
        UserJobQueryCriteria criteria = new UserJobQueryCriteria();
        criteria.setJobName(jobname);
        List<UserJob> userJobList = userJobService.queryAll(criteria);
        if (null != userJobList && !userJobList.isEmpty()) {
            UserJob userJob = userJobList.get(0);
            Price price = priceService.queryByprod(userJob.getGpuType());
            if (null != price) {
                long differ = userJob.getUpdateTime().getTime() - userJob.getCreateTime().getTime();
                long minutes = TimeUnit.MILLISECONDS.toMinutes(differ);
                double pr = price.getAmount() / 60 * userJob.getGpuNumber() * minutes;
                trade.setTotalAmount(String.format("%.2f", pr));
                String body = "GPU数量: " + userJob.getGpuNumber() + "个" + "，" + "GPU类型: " + userJob.getGpuType() + "，" + "使用时长: " + minutes + "分钟";
                trade.setBody(body);
            } else {
                trade.setTotalAmount("2000");
                trade.setBody("GPU数量 1；GPU类型：xxxx，使用时长：xx分钟");
            }
        }
        return trade;
    }

}
