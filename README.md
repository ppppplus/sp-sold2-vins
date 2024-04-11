## 代码结构

- camera_model：相机模型定义文件
- config：参数文件
- feature_tracker：前端特征处理器文件
- pose_graph：回环检测模块文件
- vins_estimator：后端位姿估计器文件

## 使用

- 首先将所有文件放入工作空间src中，编译并source，可以参照plvins.md安装依赖库
- 根据需要，修改launch文件读取的参数文件及参数文件内容（默认处理EuRoc数据集）
- 启动点线特征处理器
  - ```Go
    roslaunch feature_tracker feature_tracker.launch # 分别启动点线特征处理器
    roslaunch feature_tracker plfeature_tracker.launch # 启动点线联合特征处理器，适用于sp-sold2网络
    ```
- 启动后端位姿估计和轨迹重建
  - ```Go
    roslaunch plvins_estimator estimator.launch #运行后轨迹文件会保存到指定路径下 
    ```

对于不同的数据集，需要参照config中的文件调整参数，并在launch中指明

## 自定义前端

自定义前端包括点/线的提取、匹配方法，按照以下几个步骤进行自定义：

- 参照 feature_tracker/scripts/utils_point/superpoint/model.py 和 feature_tracker/scripts/utils_line/sold2/model.py，继承BaseExtractModel（含有extract方法）和BaseMatchModel（含有match方法），自定义提取或匹配方法
- 按照格式将继承类写入 feature_tracker/scripts/utils_point/my_point_model.py 和 feature_tracker/scripts/utils_line/my_line_model.py 的实例化函数中
- 根据自己定义的方法名和参数名，写入config中
- 此时程序将根据名字找到自定义方法，利用参数实例化并执行自定义前端特征处理器
- 预定义的前端方法包括：点提取（superpoint），点匹配（nnm, superglue）；线提取（sold2），线匹配（wunsch），以及点线联合推理的方法（sp-sold2）