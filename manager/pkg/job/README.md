# job
handle things related to job
job is created by user , then scheduled by marketmanger , then cpodmanger get the job and run it.

//后续一些思路
1. Job可以用CRD记录在集群中，这样在CPod Manager重启之后就可以知道现在有哪些Job。
  如果要获取任务的状态信息，可以先取出Job列表，然后根据Job类型和Name去读取任务的状态。（但是这一点好像也不是很有必要，只需要做一次List就可以了）
2. 既然1已经做了，为什么不把Job相关的操作用一个单独的Operater来做。
3. CPod Manager就只负责接收Job然后在集群中创建Job就可以了。（那为什么不直接创建MPIJob呢）
4. 任务的状态信息是不是应该用Monitor来做，对于各种任务类型（目前只有MPI Job）部署对应的Exporter。
