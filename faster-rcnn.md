#Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

##abstrat
<font size=4>
&emsp;&emsp;faster-rcnn是继fast-rcnn后two-stage基于region-proposal目标检测的又一大突破。rcnn检测任务可分为三个阶段：1）ss（selective search）算法提取region proposal 2）对提取的region proposals通过神经网络提取特征 3）将提取的特征经过分类器分类和回归算法对proposal的位置进行微调。fast-rcnn将rcnn的工作进行了简化变成两个阶段：1）使用ss算法提取region proposal 2对提取的proposal是经过神经网络提取特征并通过分类器进行分类和通过回归算法进行proposal位置微调。而本文直接将fast-rcnn两步目标检测任务变成了单步的可端到端训练和测试的目标检测任务。其最大的核心就是提出的RPN（Region Proposal Network）网络，使用通过rpn取代ss提取proposal步骤，并且rpn与fast-rcnn合并，共同分享使用神经网络提取的特征。<font color=	#A52A2A>这里需要首先明确的是rpn网络只是提取前景和背景的，并不是直接做分类的检测！！</font>作者在各数据集和竞赛都达到了很牛逼的结果，当然论文发表小两年了，现在无论是工业界还是学术界还都在翘首、复制和延续faster-rcnn方法。
##introduction
&emsp;&emsp;首先介绍当前基于rcnn系列的检测框架拖速度的主要原因还是在与region proposal的提取上，所以解决检测速度首先要在proposal提取阶段下手，作者分析当前诸如ss和EdgeBoxes检测方法速度都太慢（一方面这些方法在cpu上跑，肯定拖累后续的可在GPU上加速的检测网络速度，另一方面重新在GPU上实现以上算法，但其不能与后续的检测算法共享计算，速度也不会提升太多）。本文在proposal提取算法上进行了改变，通过深度神经网络（RPN）优雅地解决了proposal的计算问题，并与fast-rcnn检测框架无缝地结合在一起，共同共享卷积特征，实现proposal计算零开销。  

&emsp;&emsp;RPN网络其实是一种FCN（全卷积网络），由几层卷几层构成，通过与基于区域的检测子共同使用前阶段卷积得到的feature maps，在每个检测grid上生成对应类别分数和位置点。与之前的图像金字塔、滤波器金字塔等提取proposal方式不同，本文提出新的“anchor”方式进行proposal提取，避免了枚举图像或滤波器所有尺度和长宽比，模型在单一尺度的图像上训练和测试达到了很好的效果。  
&emsp;&emsp;为了将RPN网络和Fast-rcnn网络融合为一体的检测框架，作者提出如下轮替的训练策略:先训练微调region proposal网络，再固定region proposal网络训练微调目标检测网络。这样训练网络收敛速度加快，并使网络前部分生成的feature maps在rpn和目标检测网络中能够共享使用。

##related work
###Object proposals
* 分组超像素类方法：SS(Selective search)、CPMC、MCG 
* 基于滑动窗口方法：EdgeBoxes
###Deep Networks for Object Detection
* RCNN算法：其本质是分类器，并没有做预测目标的bounding 位置，检测其精度取决于region proposal模块
* OverFeat:假定只有单一目标，使用全连接层预测目标的坐标位置。之后全连接层被转化为卷积层用于预测多个目标。
* MutiBoxes:... 
##Faster-RCNN
&emsp;&emsp;Faster-RCNN由两个模块组成。第一个模块是用于提取region proposal的卷积神经网络，第二个模块是Fast-rcnn检测子（detector）。使用最近流行的神经网络术语“关注机制”，RPN网络告诉Fast-RCNN关注哪、往哪看。RPN和Fast-RCNN构成一个单一、一体的目标检测框架.如下图：

.....................................
![faster-rcnn](https://raw.githubusercontent.com/fire2work/Research_Paper_Book/d390fc0fa845aebd67e04d78c7e0dff6cd8dc610/assets/faster-rcnn.png)..............
![](https://raw.githubusercontent.com/fire2work/Research_Paper_Book/master/assets/faster-rcnn_train.png).............![](https://raw.githubusercontent.com/fire2work/Research_Paper_Book/master/assets/faster-rcnn_test.png)  

上图左侧为端到端训练网络的部分网络结构，右侧为测试网络的部分网络结构

* Region Proposal Networks

&emsp;&emsp;RPN的核心思想是使用CNN卷积神经网络直接产生Region Proposal，使用的方法本质上就是滑动窗口（只需在最后的卷积层上滑动一遍），因为anchor机制和边框回归可以得到多尺度多长宽比的Region Proposal。
RPN网络也是全卷积网络（FCN，fully-convolutional network），可以针对生成检测建议框的任务端到端地训练，能够同时预测出object的边界和分数。只是在CNN上额外增加了2个卷积层（全卷积层cls和reg）。
①将每个特征图的位置编码成一个特征向量（256dfor ZF and 512d for VGG）。
②对每一个位置输出一个objectness score和regressedbounds for k个region proposal，即在每个卷积映射位置输出这个位置上多种尺度（3种）和长宽比（3种）的k个（3*3=9）区域建议的物体得分和回归边界。

* anchors

&emsp;&emsp;RPN的具体流程如下：使用一个小网络在最后卷积得到的特征图上进行滑动扫描，这个滑动网络每次与特征图上n*n（论文中n=3）的"feature maps块"全连接（图像的有效感受野很大，ZF是171像素，VGG是228像素），然后映射到一个低维向量（256d for ZF / 512d for VGG），最后将这个低维向量送入到两个全连接层，即bbox回归层（reg）和box分类层（cls）。sliding window的处理方式保证reg-layer和cls-layer关联了conv5-3的全部特征空间。
reg层：预测proposal的anchor对应的proposal的（x,y,w,h）。cls层：判断该proposal是前景（object）还是背景（non-object）

作者提到，通过以上对RPN流程的描述，我们可以很自然地通过使用一个n*n(n=3)的卷积层后面接两个“同胞” 1*1的卷积层（reg和cls）来实现。

![](https://raw.githubusercontent.com/fire2work/Research_Paper_Book/master/assets/faster-rcnn_rpn.png)  
&emsp;&emsp;这里附加一段别人博客的注解：在上图中，要注意，3\*3卷积核的中心点对应原图（re-scale，源代码设置re-scale为600\*1000）上的位置（点），将该点作为anchor的中心点，在原图中框出多尺度、多种长宽比的anchors。所以，anchor不在conv特征图上，而在原图上。对于一个大小为H*W的特征层，它上面每一个像素点对应9个anchor,这里有一个重要的参数feat_stride = 16， 它表示特征层上移动一个点，对应原图移动16个像素点(看一看网络中的stride就明白16的来历了)。把这9个anchor的坐标进行平移操作，获得在原图上的坐标。之后根据ground truth label和这些anchor之间的关系生成rpn_lables，具体的方法论文中有提到，根据overlap来计算，这里就不详细说明了，生成的rpn\_labels中，positive的位置被置为1，negative的位置被置为0，其他的为-1。box\_target通过_compute_targets()函数生成，这个函数实际上是寻找每一个anchor最匹配的ground truth box， 然后进行论文中提到的box坐标的转化。http://blog.csdn.net/zhangwenjie89/article/details/52012880
  
#####1）Translation-Invariant Anchor
    
&emsp;&emsp;Anchor方式生成proposal，对图像中的目标具有平移不变性。
#####2）Muti-Scale Anchors as Regression References
&emsp;&emsp;<font color=#A52A2A>Faster-rcnn中Anchor对proposal的提取并没有显式地提取任何候选窗口，完全使用网络自身完成判断和修正。</font>
#####3）Loss Function

![](https://raw.githubusercontent.com/fire2work/Research_Paper_Book/d390fc0fa845aebd67e04d78c7e0dff6cd8dc610/assets/faster-rcnn_loss.png)

#####4）Training RPNs
&emsp;&emsp;作者通过SGD方式以端到端的方向传播算法训练RPN网络，并采用图像“中心采样”策略训练网络。每一个mini-batch采样自一幅图像中的很多正负anchors，这样能够使所有anchor都能优化rpn的损失函数，但是这样的优化会倾向于负样本（因为负样本占大多数）。于是，作者采取这样的策略：每一个mini-batch包含从一张图像中**随机提取**的256个anchor（注意，不是所有的anchor都用来训练），前景样本和背景样本均取128个，达到正负比例为1:1。如果一个图像中的正样本数小于128，则多用一些负样本以满足有256个Proposal可以用于训练。这样保证不会发生负样本倾斜问题。
具体loss的修正过程见博客：http://blog.csdn.net/qq_17448289/article/details/52871461
#####5）NMS
![](https://raw.githubusercontent.com/fire2work/Research_Paper_Book/d390fc0fa845aebd67e04d78c7e0dff6cd8dc610/assets/faster-rcnn_nms.png)
&emsp;&emsp;训练时（eg：输入600\*1000的图像），如果anchor box的边界超过了图像边界，那这样的anchors对训练loss也不会产生影响，我们将超过边界的anchor舍弃不用。一幅600\*1000的图像经过VGG16后大约为40\*60，则此时的anchor数为40*60*9，约为20k个anchor boxes，再去除与边界相交的anchor boxes后，剩下约为6k个anchor boxes，这么多数量的anchor boxes之间肯定是有很多重叠区域，因此需要使用非极大值抑制法（NMS，non-maximum suppression）将IoU＞0.7的区域全部合并，最后就剩下约2k个anchor boxes（同理，在最终检测端，可以设置将概率大约某阈值P且IoU大约某阈值T的预测框采用NMS方法进行合并，注意：这里的预测框指的不是anchor boxes）。NMS不会影响最终的检测准确率，但是大幅地减少了建议框的数量。NMS之后，我们用建议区域中的top-N个来检测（即排过序后取N个）。

* Sharing Features for RPN and Fast RCNN

下面说明在训练过程中rpn和fast--rcnn是如何实现共享卷积特征的（四步交替训练）：(Faster-R-CNN算法由两大模块组成：PRN候选框提取模块和Fast R-CNN检测模块。)

&emsp;&emsp;RPN和Fast R-CNN都是独立训练的，要用不同方式修改它们的卷积层。因此需要开发一种允许两个网络间共享卷积层的技术，而不是分别学习两个网络。注意到这不是仅仅定义一个包含了RPN和Fast R-CNN的单独网络，然后用反向传播联合优化它那么简单。原因是Fast R-CNN训练依赖于固定的目标建议框，而且并不清楚当同时改变建议机制时，学习Fast R-CNN会不会收敛。
RPN在提取得到proposals后，作者选择使用Fast-R-CNN实现最终目标的检测和识别。RPN和Fast-R-CNN共用了13个VGG的卷积层，显然将这两个网络完全孤立训练不是明智的选择，作者采用交替训练（Alternating training）阶段卷积层特征共享：  
&emsp;&emsp;第一步，我们依上述训练RPN，该网络用ImageNet预训练的模型初始化，并端到端微调用于区域建议任务；  
&emsp;&emsp;第二步，我们利用第一步的RPN生成的建议框，由Fast R-CNN训练一个单独的检测网络，这个检测网络同样是由ImageNet预训练的模型初始化的，这时候两个网络还没有共享卷积层；  
&emsp;&emsp;第三步，我们用检测网络初始化RPN训练，但我们固定共享的卷积层，并且只微调RPN独有的层，现在两个网络共享卷积层了；  
&emsp;&emsp;第四步，保持共享的卷积层固定，微调Fast R-CNN的fc层。这样，两个网络共享相同的卷积层，构成一个统一的网络。
</font>