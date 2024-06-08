强化学习（Reinforcement Learning, RL）

在人工智能的众多领域中，强化学习（Reinforcement Learning, RL）因其独特的学习机制和广泛的应用前景而备受瞩目。作为一种通过与环境交互来学习策略的机器学习方法，强化学习在自动驾驶、游戏智能体、机器人控制等领域展示了强大的潜力。在这篇文章中，我将分享我在学习强化学习技术过程中的一些心得体会，算法推演并不会说的很仔细，只是框架上的说明，我将详细介绍我在学习和实践强化学习过程中的经验和教训，希望能够为大家提供一些有价值的参考。

强化学习的典型框架：智能体（agent）通过与环境（environment）交互来学习如何在不同状态下采取行动，以最大化累积奖励（reward）。也就是说Agent看到某种行为，采取某种行为，得到了什么Reward，随后反馈给Agent。Observation跟 Action跟Reward是一种集合。例如打游戏每一回合是一个 episode，每一回合你观察到什么怪就做出不同的行为及不同金币一直到打完，你会有很多Observation跟 Action跟Reward。

![image](https://github.com/joycelai140420/Project/assets/167413809/541d9710-529b-4d08-8831-8891df6d5116)

假设游戏每一回合的同个地方遇到每一怪都是固定，但会随着你做出的Action不同也有不同的Reward，我们希望最后的total Reward 越大越好（胜利或是得到金币越多或是最快到达目的等等..），但Reward我们要怎么控制其重要性，一般范围在 [0, 1] 之间。较小的 γ 表示智能体更注重即时奖励（高尔夫一杆进洞），较大的 γ 表示智能体更注重长期奖励（和平精英伏地魔）。一个游戏打了好多回（episode），总是有好跟不好，那什么是打的好什么是打得不好，我们就需要 baseline做标准。

一下这张图就是S可以想象成每一个Observation，a是Action，A1是一个episode，G可以说是Reward集合，b就是baseline。

![1717580191362](https://github.com/joycelai140420/Project/assets/167413809/ab0eaabc-6bce-423a-936e-903be5f7221d)

假设（游戏动作只有上下），那个动作{S1,上,S2,下,S3,上,....St,下}叫trajectory，每个trajectory里面都不一样有可能是{S1,上,S2,上,S3,上,....St,上}等等，那么一个trajectory会发生几率是多少？（这里也要注意到是可能S1,下一个画面接的也不一定是S2）,因为trajectory是一个随机需要算机率，所以会发生什么Reward还是需要算机率。所以期望值就是穷举所有trajectory每一个机率，那么假设游戏一直都没死的trajectory出现的机率就最大，那么一下就死的trajectory出现的机率就最小。那么我们算出甲trajectory出现的机率再算这个甲trajectory的total Reward，然后把所有甲乙丙丁的total Reward做一个权重算出甲的期望值，当然我们希望期望值是越大越好。（穷举什么什么得到最小或最大等用词就需要用到Gradient）这里我们需要期望值是越大越好，要用max，这就是GradientPolicy。

也就是说你把你的 theta 加上你的 gradient 這一項，那當然前面要有個 learning rate learning rate 其實也是要調的，你要用 ADAM、rmsprop 等等，去調一下，那在實際上做的時候，要套下面這個公式， 首先你要先收集一大堆的 s 跟 a 的 pair，你還要知道這些 s 跟 a，如果實際上在跟環境互動的時候 你會得到多少的 reward， 所以這些資料，你要去收集起來，這些資料怎麼收集呢？ 你就要拿你的 agent，它的參數是 theta，去跟環境做互動， 也就是你拿你現在已經 train 好的那個 agent，先去跟環境玩一下，先去跟那個遊戲互動一下， 那互動完以後，你就會得到一大堆遊戲的紀錄會記錄說，今天先玩了第一場 在第一場遊戲裡面，我們在 state s1，採取 action a1，在 state s2，採取 action a2，要記得說其實今天玩遊戲的時候，是有隨機性的 所以你的 agent 本身是有隨機性的，所以在同樣 state s1，不是每次都會採取 a1，所以你要記錄下來，在 state s1，採取 a1，在 state s2，採取 a2，然後最後呢 整場遊戲結束以後，得到的分數，是 R of tao1，那你还會 sample 到另外一筆 data，也就是另外一場遊戲，在另外一場遊戲裡面 你在第一個 state 採取這個 action，在第二個 state 採取這個 action，在第二個遊戲畫面採取這個 action，然後你 sample 到的，你得到的 reward 是 R of tao2，你有了這些東西以後，你就去把這邊你 sample 到的東西，帶到這個 gradient 的式子裡面，把 gradient 算出來 也就是說你會做的事情是，把這邊的每一個 s 跟 a 的 pair，拿進來，算一下它的 log probability，你計算一下，在某一個 state，採取某一個 action 的 log probability，然後對它取 gradient。然後這個 gradient 前面會乘一個 weight，這個 weight 是什麼？這個 weight 就是会加权算出這場遊戲在某一個 state，採取某一個 action 的 reward，你有了這些以後，你就會去 update 你的 model， 你 update 完你的 model 以後，你回過頭來要重新再去收集你的 data，然後再去收集你的 data，再 update model...那這邊要注意一下，一般 policy gradient，你 sample 的 data 就只會用一次，你把這些 data sample 起來，然後拿去 update 參數，這些 data 就丟掉了 再重新 sample data，才能夠再重新去 update 參數。

![1717640846325](https://github.com/joycelai140420/Project/assets/167413809/cd716c53-753f-481f-914a-d55bc40f2d2d)

那么我们引入baseline是为什么呢？在实作上，可能这场游戏得到最后的结果是不好，不代表里面所有的state採取某个 action都是不好，反之，这场游戏得到最后的结果是好，不代表里面所有的state採取某个 action都是好的，那么我就要引入baseline，在强化学习中，基线是一种用来减少算法在训练过程中不稳定性的技巧。它可以帮助我们的算法更稳定、更快速地找到最好的策略。你可以把基线想象成一个参考点，我们通过它来评估每个动作到底是比平均水平好还是不好。基线的简单计算方法一个简单的基线计算方法是计算多个回合的平均回报。具体来说，就是运行多次游戏，然后计算这些游戏的平均得分，把这个平均得分作为基线。

这个步驟如下：
    
    运行多个游戏回合：我们让智能体玩多次游戏（例如10次）。
    
    计算每个回合的总回报：对于每个游戏回合，我们计算它的总得分。
    
    计算平均回报：把所有回合的总得分相加，然后除以回合的数量，得到平均得分，这就是我们的基线。

接下来我们这个例子中实作（RL_Simple_baseline_PPO.ipynb），我们让agent玩了10次游戏，计算每次游戏的总得分，并取这些得分的平均值作为基线。基线帮助我们在更新策略时减少波动，使训练过程更加平稳。在这个例子中，我们没有显式地在代码中实现减去基线的步骤，这是因为我们使用的 PPO 算法已经在内部处理了这个问题。
    tips:
    选择γ=0.99，使得智能体在很大程度上重视未来的奖励，同时不过分忽略当前的奖励。在许多实际应用中，发现γ=0.99能在各种不同类型的任务中表现良好，因此成为一个常见的默认选择。如果任务需要长期策略（例如投资、规划），可以选择较大的 γ（接近 1）。如果任务更关注短期收益（例如即时反应），可以选择较小的γ（接近 0）。
    
https://github.com/joycelai140420/Project/assets/167413809/48f2def0-0c08-4ff5-9c0a-c00c95d8550f


假设我们要跟环境互动的那个agent跟我们要learn agent是同一个的话,这个叫做on-policy。反之，要跟环境互动的那个agent跟我们要learn agent不是同一个的话,这个叫做off-policy。简单来说就是这个agent是边学边玩的叫on-policy。反之，这个agent是透过别人玩来学习的叫off-policy。前面介绍的都是on-policy，现在我们开始介绍一下off-policy。

我们就先从 Q-learning 的简介开始说，那我们说 Q-learning 这种方法，它是 value-based 的方法， 在value based 的方法里面我们 learn 的并不是 policy，我们并不是直接 learn policy， 我们要 learn 的是一个 critic，critic 并不直接采取行为，它想要做的事情是评价现在的行为有多好或者是有多不好。那如果在玩 Atari 游戏的话，state s 是某一个画面 ，看到某一个画面，某一个 state s 的时候，接下来一直玩到游戏结束 ，看累积的 reward 的期望值有多大。

Q-Learning通过不断更新状态-动作值函数（Q值）来逼近最优Q值函数。其更新公式为：

![1717760107800(1)](https://github.com/joycelai140420/Project/assets/167413809/2514ac59-4aac-4b57-b095-699a177effd9)

Q-Learning的基本概念
    状态（State, s）：环境的当前配置。
    
    动作（Action, a）：智能体在每个状态下可以采取的动作。

    奖励（Reward, r）：智能体在采取某个动作后从环境中获得的反馈。

    Q值（Q-value, Q(s, a)）：在状态 𝑠 下采取动作𝑎 后的期望累计奖励。   

其中：
    α 是学习率，控制更新的步长。
    γ 是折扣因子，决定未来奖励的重要性。
    r 是即时奖励。
    s′是采取动作 
    𝑎 后的下一状态。
    max𝑎′𝑄(𝑠′,𝑎′)表示在状态𝑠′下选择最优动作的Q值。

Q-Learning的步骤
    初始化Q值表 𝑄(𝑠,𝑎)    为任意值（通常为0）。
    
    重复以下步骤直到收敛：

        在当前状态 𝑠下选择动作 𝑎（根据某种策略，如 ϵ-贪婪策略、Softmax...）。
        执行动作 𝑎，观察奖励 𝑟 和下一个状态𝑠′ 。
        更新Q值 𝑄(𝑠,𝑎)。
        更新状态 𝑠←𝑠′。

Q-Learning算法最早由Christopher J.C. Watkins在1989年提出，并在1992年的论文中详细描述。论文题目是："Q-Learning" by Christopher J.C. Watkins and Peter Dayan (1992)。Watkins和Dayan在这篇论文中提出了一种新的强化学习算法，称为Q-Learning。该算法能够在不确定和变化的环境中找到最优策略，而不需要知道环境的状态转移模型。他们通过理论分析和实验验证了Q-Learning的有效性和收敛性。

以下是使用ϵ-贪婪策略的Q-Learning，可以參考RL_Q-Learning.ipynb，做了测试训练后的策略我选择episode 125，以下是画面

这是第一个回合的视频。agent刚开始学习，策略可能还很不成熟，一直不断探索无章节乱走，最后就卡在上面一直撞上面。

https://github.com/joycelai140420/Project/assets/167413809/68dc00b1-cfda-4cd7-a6d0-7d7f32d0279c

这是第二个回合的视频。agent可能还在探索环境，但是比起上面，其动作可能显得比较随机

https://github.com/joycelai140420/Project/assets/167413809/a37deb68-1cfd-498d-8c29-7d012a13dc28

这是第八个回合的视频。随着回合的增加，agent逐渐学习到了更好的策略，可能已经能够在环境中存活更长时间但可能会很凑巧到达目标。

https://github.com/joycelai140420/Project/assets/167413809/e166c665-097c-4143-93ac-06e81166fa7a

这是第二十七个回合的视频。到这个回合，agent可能已经学到了一些有效的策略，能够避开陷阱并朝着目标移动，但可能还不完全稳定。

https://github.com/joycelai140420/Project/assets/167413809/a14e0e66-80be-4fff-a698-531859cd4b15

这是第六十四个回合的视频。在这个回合中，你看到小人在上上下下，最终走到终点。这表明agent已经学到了一些有效的策略，并能在一定程度上成功完成任务。上上下下的动作可能是agent在局部区域内进行微调，寻找最佳路径。

https://github.com/joycelai140420/Project/assets/167413809/04732ef3-43b6-4bbd-8a45-83e4bca3f3aa

这是第125个回合的视频。智能体已经有了较为稳定的策略，一打开就能看到它已经走到了终点。这表明智能体已经学会了如何高效地完成任务，在环境中表现得非常好。

https://github.com/joycelai140420/Project/assets/167413809/97fcb0aa-ddc9-41d4-9253-a646b419951a
