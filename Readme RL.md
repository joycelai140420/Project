强化学习（Reinforcement Learning, RL）

在人工智能的众多领域中，强化学习（Reinforcement Learning, RL）因其独特的学习机制和广泛的应用前景而备受瞩目。作为一种通过与环境交互来学习策略的机器学习方法，强化学习在自动驾驶、游戏智能体、机器人控制等领域展示了强大的潜力。在这篇文章中，我将分享我在学习强化学习技术过程中的一些心得体会，算法推演并不会说的很仔细，只是框架上的说明，我将详细介绍我在学习和实践强化学习过程中的经验和教训，希望能够为大家提供一些有价值的参考。

强化学习的典型框架：智能体（agent）通过与环境（environment）交互来学习如何在不同状态下采取行动，以最大化累积奖励（reward）。也就是说Agent看到某种行为，采取某种行为，得到了什么Reward，随后反馈给Agent。Observation跟 Action跟Reward是一种集合。例如打游戏每一回合是一个 episode，每一回合你观察到什么怪就做出不同的行为及不同金币一直到打完，你会有很多Observation跟 Action跟Reward。

![image](https://github.com/joycelai140420/Project/assets/167413809/541d9710-529b-4d08-8831-8891df6d5116)

假设游戏每一回合的同个地方遇到每一怪都是固定，但会随着你做出的Action不同也有不同的Reward，我们希望最后的total Reward 越大越好（胜利或是得到金币越多或是最快到达目的等等..），但Reward我们要怎么控制其重要性，一般范围在 [0, 1] 之间。较小的 γ 表示智能体更注重即时奖励（高尔夫一杆进洞），较大的 γ 表示智能体更注重长期奖励（和平精英伏地魔）。一个游戏打了好多回（episode），总是有好跟不好，那什么是打的好什么是打得不好，我们就需要 baseline做标准。

一下这张图就是S可以想象成每一个Observation，a是Action，A1是一个episode，G可以说是Reward集合，b就是baseline。

![1717580191362](https://github.com/joycelai140420/Project/assets/167413809/ab0eaabc-6bce-423a-936e-903be5f7221d)

假设（游戏动作只有上下），那个动作{S1,上,S2,下,S3,上,....St,下}叫trajectory，每个trajectory里面都不一样有可能是{S1,上,S2,上,S3,上,....St,上}等等，那么一个trajectory会发生几率是多少？（这里也要注意到是可能S1,下一个画面接的也不一定是S2）,因为trajectory是一个随机需要算机率，所以会发生什么Reward还是需要算机率。所以期望值就是穷举所有trajectory每一个机率，那么假设游戏一直都没死的trajectory出现的机率就最大，那么一下就死的trajectory出现的机率就最小。那么我们算出甲trajectory出现的机率再算这个甲trajectory的total Reward，然后把所有甲乙丙丁的total Reward做一个权重算出甲的期望值，当然我们希望期望值是越大越好。（穷举什么什么得到最小或最大等用词就需要用到Gradient）这里我们需要期望值是越大越好，要用max，这就是GradientPolicy。


假设我们要跟环境互动的那个agent跟我们要learn agent是同一个的话,这个叫做on-policy。反之，要跟环境互动的那个agent跟我们要learn agent不是同一个的话,这个叫做off-policy。简单来说就是这个agent是边学边玩的叫on-policy。反之，这个agent是透过别人玩来学习的叫off-policy。

然后我们需要知道
