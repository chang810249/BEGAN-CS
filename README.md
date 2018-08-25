# Escaping from Collapsing Modes in a Constrained Space
[Chia-Che Chang*](http://chang810249.github.io), [Chieh Hubert Lin*](https://hubert0527.github.io), [Che-Rung Lee](http://www.cs.nthu.edu.tw/~cherung/), [Da-Cheng Juan](https://ai.google/research/people/DaChengJuan), [Wei Wei](https://ai.google/research/people/105672), [Hwann-Tzong Chen](http://www.cs.nthu.edu.tw/~htchen/)

The authors' TensorFlow implementation of ECCV'18 paper, "[Escaping from Collapsing Modes in a Constrained Space](https://arxiv.org/abs/1808.07258)".

** This is not an official Google product **

## PCA Visualization
<p align="center">
    <img src="https://i.imgur.com/0T0o6Gw.png"/>
</p>

## Model Architecture
<p align="center">
    <img src=https://i.imgur.com/nzS8zmV.png/>
</p>

## Results

**Train on CelebA dataset**
<p align="center">
    <img src="https://i.imgur.com/gnNvYwK.png" width="100%"/>
    <img src="https://i.imgur.com/IqLtAu3.png" width="100%"/>
</p>

<hr size="50">

**Train on 1/10 CelebA dataset**
<p align="center">
    <img src="https://i.imgur.com/GT7NCxn.png" width="100%"/>
    <img src="https://i.imgur.com/g04PzTm.png" width="100%"/>
</p>

<hr size="50">

**Selected disentangled representations of BEGAN-CS**
<p align="center">
    <img src="https://i.imgur.com/V72GFu6.png" width="100%"/>
</p>

<hr size="50">

**Two-dimensional combinations of disentangled representations**
<p align="center">
    <img src="https://i.imgur.com/P1T72jd.png" width="100%"/>
</p>

<hr size="50">

**Experimental results on the synthetic dataset**
<p align="center">
    <img src="https://i.imgur.com/8KpvIWd.png" width="100%"/>
</p>

<hr size="50">

**Image reconstruction results**
<p align="center">
    <img src="https://i.imgur.com/HtBrw1b.png" width="100%"/>
</p>

<hr size="50">

**Interpolation**
<p align="center">
    <img src="https://i.imgur.com/MLrmt08.png" width="100%"/>
</p>

<!--
<p align="center">
    <img src="https://i.imgur.com/X1NwqoV.png" width="100%"/>
    <center><h3> z*-search </h3></center>
</p>
-->

<!--
<p align="center">
    <img src="https://i.imgur.com/7ldxXPb.jpg" class="center"/>
    <img src="https://i.imgur.com/rE3lcoM.jpg" class="center"/>
    <center><h3> Disentangled representations of BEGAN-CS across 64 dimensions along each axis in latent space Z. </h3></center>
</p>

-->

### Reference
> 1. [syentic dataset](https://github.com/akashgit/VEEGAN)
> 2. [BEGAN](https://github.com/carpedm20/BEGAN-tensorflow)
> 3. [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)
