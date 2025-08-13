格式说明：
1. 表达式格式： <name>::<表达式;num 数据1名称 数据1编号 数据2名称 数据2编号...>::<均值平滑>::<指数平滑>::<中性化方法>::<标准化方法>::<InSample起始时间>::<OutSample起始时间>
2. 数据说明：
	TCFBaseClean{1, 1}	Ret
	TCFBaseClean{1, 2}	open
	TCFBaseClean{1, 3}	high
	TCFBaseClean{1, 4}	low
	TCFBaseClean{1, 5}	close
	TCFBaseClean{1, 6}	vol
	TCFBaseClean{1, 7}	oi
	TCFBidAskPrice{1,1}	mean(a-b)
	TCFBidAskPrice{1,2}	(2p-a-b)/(a-b), mean
	TCFBidAskPrice{1,3}	(2p-a-b)/(a-b) * vol, sum
	TCFBidAskPrice{1,4}	vwap (sum(p * vol)/sum(vol))
	TCFBidAskPrice{1,5}	p vola(mean(abs(mean(p) -p)))
	TCFBidAskPrice{1,6}	(a-b) vola
	TCFBidAskPrice{1,7}	a vola
	TCFBidAskPrice{1,8}	b vola
	TCFBidAskPrice{1,9}	delta(a) * va, sum
	TCFBidAskPrice{1,10}	delta(b) * vb, sum
	TCFBidAskPrice{1,11}	(delta(a)-delta(b)) * vol, sum
	TCFBidAskPrice{1,12}	sum(vol)

3. 部分操作符说明：
	ts_detrend( data, n ): 		当前分钟数据减去n日内当前分钟的均值
	ts_norm( data, t ): 		当前分钟数据减去过去可交易的t分钟的均值，再除过去可交易t分钟标准差
	sign( data ): 				取信号的符号量，值为+-1或0
	ts_sub_mean( data, t ): 	ts_norm的分子部分
	ts_ret( data, t ): 			当前分钟数据除t分钟之前数据再减1，Return的计算逻辑
	ts_corr( data1, data2, t ):	data1和data2在过去t分钟内求相关性数值，赋值到当前时刻
	mul_p：						等同于mul，第二个变量为具体数字