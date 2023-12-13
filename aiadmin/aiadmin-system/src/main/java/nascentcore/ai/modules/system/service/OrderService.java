package nascentcore.ai.modules.system.service;

import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.modules.system.domain.AliPayRes;
import nascentcore.ai.modules.system.domain.vo.OrderQueryCriteria;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import nascentcore.ai.modules.system.domain.Order;
import nascentcore.ai.utils.PageResult;

import java.util.List;


public interface OrderService extends IService<Order> {

    /**
     * 查询数据分页
     *
     * @param criteria 条件
     * @param page     分页参数
     * @return PageResult
     */
    PageResult<Order> queryAll(OrderQueryCriteria criteria, Page<Object> page);

    /**
     * 查询所有数据不分页
     *
     * @param criteria 条件参数
     * @return List<OrderDto>
     */
    List<Order> queryAll(OrderQueryCriteria criteria);

    /**
     * 查询订单单个数据
     *
     * @param criteria 条件参数
     * @return List<OrderDto>
     */
    Order queryOne(OrderQueryCriteria criteria);

    /**
     * 通过订单号查询订单
     *
     * @param outTradeNo 订单号
     * @return OrderDto
     */
    Order querybyNo(String outTradeNo);

    /**
     * 创建
     *
     * @param resources /
     */
    void create(Order resources);

    /**
     * 编辑
     *
     * @param resources /
     */
    void update(Order resources);

    /**
     * 根据job获取支付信息
     *
     * @param jobname /
     */
    AliPayRes getPayInfo(String jobname);

}
