---
title: 图片边缘检测并且拟合直线
date: 2019-1-19
categories: [project, Computer Vision]
tags: [Edge detection]
mathjax: true
---

前段时间做了一个简单的小项目，需求是测量图片中布的褶皱角度，第一次做CV的东西，决定用这个blog记录一下。 项目源码在:[edge-detection
](https://github.com/zhuo931077127/edge-detection "edge-detection")

先看看传统的边缘检测方法的效果:

![你想输入的替代文字](tra.png)

第一张图是原始图，由于本项目要求竟要求图片上半部分的褶皱角度，所以仅考虑上半部分的背景干扰，可以看出高斯，梯度，非极大抑制这三种方法都无法有效的排除干扰。
然后我们用canny算法试试。  
步骤是先把图片转化为灰度图，然后用canny算子做边缘检测。
代码如下：  
{% codeblock %}
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 310)  # apertureSize参数默认其实就是3  # 50 310
    # cv.imshow("edges", edges)
    edge = Image.fromarray(edges)
    edge.save("edge.jpeg")
{% endcodeblock %}

结果如下:  

![你想输入的替代文字](edge.jpeg)

霍夫线性变换拟合直线:

{% codeblock %}
 lines = cv.HoughLines(edges, 1, np.pi / 180, 68)  # 68
    # l1 = lines[:, 0, :]
    # print(l1)
    mink = float('inf')
    maxk = -float('inf')
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho  # 代表x = r * cos（theta）
        y0 = b * rho  # 代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        print("x1: %s, y1:%s, x2:%s, y2:%s" % (x1, y1, x2, y2))
        k = (y2 - y1) / (x2 - x1)
        if k > maxk:
            maxk = k
            xmax1 = x1
            ymax1 = y1
            xmax2 = x2
            ymax2 = y2
            lineMax = line
        if k < mink:
            mink = k
            xmin1 = x1
            ymin1 = y1
            xmin2 = x2
            ymin2 = y2
            lineMin = line
    cv.line(image, (xmax1, ymax1), (xmax2, ymax2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。
    cv.line(image, (xmin1, ymin1), (xmin2, ymin2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。

{% endcodeblock %}

值得注意的是，拟合直线过程中HoughLines会发现很多直线，因此，我选择了斜率最大和最小的两条直线做为最终的直线。画在图上的话就是酱紫的结果:

![你想输入的替代文字](line.png)

所以：总的过程可以概括如下:

![你想输入的替代文字](final.jpeg)

现在想这么弱智的项目居然还花了几天时间做，我是真滴蠢哦（T-T）