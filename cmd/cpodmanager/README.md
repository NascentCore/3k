# cpodmanager

## cpodmanager usage
cpodmanager [ USER_ID ]  [ CPOD_ID ]  [ BASE_URL]

example :
./cpodmanager user0001  xxxxxxx   http://sxwl.ai/controlplane

USER_ID 可以通过在算想未来官方网站上注册获得，是客户身份的唯一标识。
CPOD_ID 必须能够唯一标识cpodmanager所处的K8S集群，最好通过集群本身的硬件序列号等信息经过HASH得到。
如果在多个集群当中采用相同的CPOD_ID运行cpodmanager会导致严重的问题！！！
BASE_URL 代表cpodmanager与外界通信的地址。通信的目的 1 上报集群状态 2 获取要在此集群中执行的Job信息 
 