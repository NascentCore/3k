import ReactECharts from 'echarts-for-react';

const UsageChart: React.FC = () => {
  const chartOption = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: ['03-01', '03-02', '03-03'],
      axisLabel: {
        interval: 0,
        rotate: 0,
      },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: {
        formatter: '${value}',
      },
    },
    series: [
      {
        name: 'Usage',
        type: 'bar',
        data: [0, 0, 0],
        itemStyle: {
          color: '#1890ff',
        },
      },
    ],
  };

  return (
    <ReactECharts
      option={chartOption}
      style={{ height: '280px', width: '90%' }}
      opts={{ renderer: 'svg' }}
    />
  );
};

export default UsageChart; 