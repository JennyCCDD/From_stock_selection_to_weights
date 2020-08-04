# From_stock_selection_to_weights

This is what I did for a contest in 2020/08/2. 

In this contest, participates had to get the updated data from GRPC and upload their weights for each stock.

So, I learned how to write GRPC in python from this [dewei_zhang的grpc-pytho](https://space.bilibili.com/376995746/channel/detail?cid=124886n)

To learn something from the sample data given, I refered this link to try to get new factors [聚宽-基于遗传规划自动挖掘因子](https://www.joinquant.com/view/community/detail/6e594923d168b1592e8737c88988d91e?type=1). However, the new factors are not good.

![A example factor](https://github.com/JennyCCDD/From_stock_selection_to_weights/blob/master/to-show/6.png)

Also, I used 国泰君安 Alpha191 to generate technical factors and use LightBGM to predict the daily return using these factors. However, with or without factors selection, the performances are disappointing. 

Then, I tried [online portfolio selection](https://github.com/Marigold/universal-portfolios). In sample result was good but I didn't have the time to use it in the real contest.

![Alt](https://github.com/JennyCCDD/From_stock_selection_to_weights/blob/master/to-show/result.png#pic_center =30x30)

