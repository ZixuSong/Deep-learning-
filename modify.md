<a name="5.1.4"></a>
### 5.1.4 Bayesian Models

**[ICML-2016]**
[Dropout as a bayesian approximation: Representing model uncertainty in deep learning](http://proceedings.mlr.press/v48/gal16.pdf).
<br>
**Authors:** Yarin Gal , Zoubin Ghahramani
<br>
**Institution:** University of Cambridge
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>


**[NeurIPS-2017]**
[Simple and scalable predictive uncertainty estimation using deep ensembles](https://arxiv.org/pdf/1612.01474.pdf). 
<br>
**Authors:** Balaji Lakshminarayanan , Alexander Pritzel , Charles Blundell
<br>
**Institution:** DeepMind
> <details>
> <summary></summary>
> <p style="text-align:left">
> use multiple semantic dense
representations as the target label to train the OOD detection network.
> </p>
> </details>


**[NeurIPS-2019]**
[Practical deep learning with bayesian principles](https://arxiv.org/pdf/1906.02506.pdf)
**Authors:** Kazuki Osawa, Siddharth Swaroop, Anirudh Jain, Runa Eschenhagen, Richard E. Turner, Rio Yokota, Mohammad Emtiyaz Khan
**Institution:** Tokyo Institute of Technology; University of Cambridge; Indian Institute of Technology (ISM); University of Osnabrück; RIKEN Center for AI Project
> <details>
> <summary>Large-scale pre-trained transformers significantly improve near-OOD tasks</summary>
> <p style="text-align:left">
> This work explores the effectiveness of large-scale pre-trained transformers, especially when few-shot outlier exposure is available. It also shows that the pre-trained multi-modal image-text transformers CLIP is also effective on OOD detection if using the names of outlier classes as candidate text labels.
> </p>
> </details>


**[NeurIPS-2018]**
[Predictive Uncertainty Estimation via Prior Networks](https://arxiv.org/pdf/1802.10501.pdf)
<br>
**Authors:** Andrey Malinin, Mark Gales
<br>
**Institution:** University of Cambridge 
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>


**[NeurIPS-2019]**
[Reverse kl-divergence training of prior networks: Improved uncertainty and adversarial robustness](https://arxiv.org/pdf/1905.13472.pdf)
<br>
**Authors:** Andrey Malinin, Mark Gales
<br>
**Institution:** Yandex; University of Cambridge
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>


**[NeurIPS-2020]**
[Towards maximizing the representation gap between in-domain & out-of-distribution examples](https://arxiv.org/pdf/2010.10474.pdf)
<br>
**Authors:** Jay Nandy, Wynne Hsu, Mong Li Lee
<br>
**Institution:** National University of Singapore
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>


<br>

[Back to Top](#top)

<br>


<a name="5.1.5"></a>
### 5.1.5 Large-scale OOD Detection

**[CVPR-2021]**
[Mos: Towards scaling out-of-distribution detection for large semantic space](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_MOS_Towards_Scaling_Out-of-Distribution_Detection_for_Large_Semantic_Space_CVPR_2021_paper.pdf)
<br>
**Authors:** Rui Huang, Yixuan Li
<br>
**Institution:** University of Wisconsin-Madison
> <details>
> <summary>Using the class conditional Gaussian distributions with respect to low- and upper-level features.</summary>
> <p style="text-align:left">
> The model estimates standard class-conditional Gaussian distribution on intermediate activations to detect OOD samples. It makes use of Mahalanobis distance therefore it can be also interpreted as distance-based method.
> </p>
> </details>


**[arXiv-2021]**
[Exploring the limits of out-of-distribution detection](https://arxiv.org/pdf/2106.03004.pdf)
<br>
**Authors:** Stanislav Fort, Jie Ren, Balaji Lakshminarayanan
<br>
**Institution:** Stanford University; Google Research
> <details>
> <summary>Using likelihood ratios to cancel out background influence.</summary>
> <p style="text-align:left">
> This work finds the likelihood score is heavily affected by background, so likelihood ratios are used to cancel out background influence. The Likelihood Ratio (LR) is the likelihood that a given test result would be expected in a patient with the target disorder compared to the likelihood that that same result would be expected in a patient without the target disorder.
> </p>
> </details>



**[arXiv-2020]**
[Pretrained transformers improve out-of-distribution robustness](https://arxiv.org/pdf/2004.06100.pdf)
br>
**Authors:** Dan Hendrycks, Xiaoyuan Liu1, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, Dawn Song
<br>
**Institution:** UC Berkeley; Shanghai Jiao Tong University; University of Chicago
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>



**[arXiv-2021]**
[Oodformer: Out-of-distribution detection transformer](https://arxiv.org/pdf/2107.08976.pdf)
<br>
Authors: Rajat Koner, Poulami Sinhamahapatra, Karsten Roscher, Stephan Günnemann, Volker Tresp 
<br>
**Institution:** Ludwig Maximilian University; Technical University; Fraunhofer, IKS; Siemens AG
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>


<br>

[Back to Top](#top)

<br>

<a name="5.2"></a>
## 5.2 Density-based Methods
**[ICLR-2018]**
[Deep autoencoding gaussian mixture model for unsupervised anomaly detection](https://openreview.net/pdf?id=BJJLHbb0-)
<br>
**Authors:** Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, Haifeng Chen
<br>
**Institution:** NEC Laboratories America; Washington State University
> <details>
> <summary></summary>
> <p style="text-align:left">
> This work shows that even powerful neural generative models often assign higher probabilities to out-of-distribution test examples than to in-distribution test examples.
> </p>
> </details>



**[CVPR-2019]**
[Latent space autoregression for novelty detection](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.pdf)
<br>
**Authors:** Davide Abati, Angelo Porrello, Simone Calderara, Rita Cucchiara
<br>
**Institution:** University of Modena and Reggio Emilia
> <details>
> <summary></summary>
> <p style="text-align:left">
> We discover that such conventional novelty detection schemes are also vulnerable to blurred images.
> </p>
> </details>



**[NeurIPS-2018]**
[Generative probabilistic novelty detection with adversarial autoencoders](https://arxiv.org/pdf/1807.02588.pdf)
<br>
**Authors:** Stanislav Pidhorskyi, Ranya Almohsen, Donald A. Adjeroh, Gianfranco Doretto
<br>
**Institution:** West Virginia University
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>



**[in Joint european conference on machine learning and knowledge discovery in databases-2018]**
[Image anomaly detection with generative adversarial networks](https://openreview.net/pdf?id=S1EfylZ0Z)
<br>
**Authors:** Lucas Deecke, authorRobert Vandermeulen, Lukas RuffStephan Mandt, Marius Kloft
<br>
**Institution:** University of Edinburgh; TU Kaiserslautern; Hasso Plattner Institute; University of California
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>



**[TPAMI-2020]**
[Normalizing flows: An introduction and review of current methods](https://arxiv.org/pdf/1908.09257.pdf)
<br>
**Authors:** Ivan Kobyzev, Simon J.D. Prince, Marcus A. Brubaker
<br>
**Institution:** Borealis AI
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>


**[CVPR-2018]**
[Adversarially learned one-class classifier for novelty detection](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf)
<br>
**Authors:** Mohammad Sabokrou, Mohammad Khalooei, Mahmood Fathy, Ehsan Adeli
<br>
**Institution:** Institute for Research in Fundamental Sciences; Amirkabir University of Technology; Stanford University
> <details>
> <summary></summary>
> <p style="text-align:left">
> It uses a residual flow, a novel flow architecture that learns the residual distribution from a base Gaussian distribution.
> </p>
> </details>



**[NeurIPS-2018]**
[A simple unified framework for detecting out-of-distribution samples and adversarial attacks](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)
<br>
**Authors:** Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin
<br>
**Institution:** Korea Advanced Institute of Science and Technology (KAIST); University of Michigan; Google Brain; AItrics
> <details>
> <summary></summary>
> <p style="text-align:left">
> Conclusion is that flows do not represent images based on their semantic contents, but rather directly encode their visual appearance.
> </p>
> </details>



**[CVPR-2020]**
[Deep residual flow for out of distribution detection](?)
<br>
**Authors:** Ev Zisselman, Aviv Tamar
<br>
**Institution:** Technion
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>



**[NeurIPS-2018]**
[Glow: Generative flow with invertible 1x1 convolutions](https://arxiv.org/pdf/1807.03039.pdf)
<br>
**Authors:** Diederik P. Kingma, Prafulla Dhariwal
<br>
**Institution:** OpenAI
> <details>
> <summary></summary>
> <p style="text-align:left">
> 
> </p>
> </details>



**[ICML-2016]**
[Pixel recurrent neural networks](http://proceedings.mlr.press/v48/oord16.pdf)
<br>
**Authors:** Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu
<br>
**Institution:** Google DeepMind
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


**[NeurIPS-2018]**
[Do deep generative models know what they don’t know?](https://arxiv.org/pdf/1810.09136.pdf)
<br>
**Authors:** Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, Balaji Lakshminarayanan
<br>
**Institution:** DeepMind
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


**[arXiv-2018]**
[Waic, but why? generative ensembles for robust anomaly detection](https://arxiv.org/pdf/1810.01392.pdf)
<br>
**Authors:** Hyunsun Choi, Eric Jang, Alexander A. Alemi
<br>
**Institution:** Google Inc.
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


**[NeurIPS-2020]**
[Why normalizing flows fail to detect out-of-distribution data](https://arxiv.org/pdf/2006.08545.pdf)
<br>
**Authors:** Polina Kirichenko, Pavel Izmailov, Andrew Gordon Wilson
<br>
**Institution:** New York University
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


**[NeurIPS-2019]**
[Likelihood ratios for out-of-distribution detection](https://arxiv.org/pdf/1906.02845.pdf)
<br>
**Authors:** Jie Ren, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark A. DePristo, Joshua V. Dillon, Balaji Lakshminarayanan
<br>
**Institution:** Google Research; DeepMind
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


**[ICLR-2020]**
[Input complexity and out-of-distribution detection with likelihood-based generative models](https://arxiv.org/pdf/1909.11480.pdf)
<br>
**Authors:** Joan Serra, David Alvarez, Vicenc¸ Gomez, Olga Slizovskaia, Jose F. Nunez, Jordi Luque
<br>
**Institution:** Dolby Laboratories; Telefonica Research;  Universitat Politecnica de Catalunya; Universitat Pompeu Fabra
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


**[NeurIPS-2020]**
[Likelihood regret: An out-ofdistribution detection score for variational auto-encoder](https://arxiv.org/pdf/1906.02845.pdf)
<br>
**Authors:** Joan Serra, David Alvarez, Vicenc¸ Gomez, Olga Slizovskaia, Jose F. Nunez, Jordi Luque
<br>
**Institution:** Dolby Laboratories; Telefonica Research;  Universitat Politecnica de Catalunya; Universitat Pompeu Fabra
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


<br>

[Back to Top](#top)

<br>

<a name="5.3"></a>
## 5.3 Distance-based Methods
**[NeurIPS-2018]**
[A simple unified framework for detecting out-of-distribution samples and adversarial attacks](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)
<br>
**Authors:** Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin
<br>
**Institution:** Korea Advanced Institute of Science and Technology (KAIST); University of Michigan; Google Brain; AItrics
> <details>
> <summary></summary>
> <p style="text-align:left">
> Conclusion is that flows do not represent images based on their semantic contents, but rather directly encode their visual appearance.
> </p>
> </details>


**[NeurIPS-2019]**
[Likelihood ratios for out-of-distribution detection](https://arxiv.org/pdf/1906.02845.pdf)
<br>
**Authors:** Jie Ren, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark A. DePristo, Joshua V. Dillon, Balaji Lakshminarayanan
<br>
**Institution:** Google Research; DeepMind
> <details>
> <summary>Using energy scores instead of softmax scores to conveniently achieve good results.</summary>
> <p style="text-align:left">
> Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. The paper shows that energy can conveniently replace softmax confidence for any pre-trained neural network, and proposes an energy-bounded learning objective to fine-tune the network.
> </p>
> </details>


**[arXiv-2021]**
[A simple fix to mahalanobis distance for improving near-ood detection](https://arxiv.org/abs/1905.10628)
<br>
**Authors:** Techapanurak, Engkarat and Suganuma, Masanori and Okatani, Takayuki
<br>
**Institution:** Tohoku University; RIKEN
> <details>
> <summary> Using scaled cosine similarity between test sample features and class features to determine OOD samples. </summary>
> <p style="text-align:left">
> The first work employs softmax of scaled cosine similarity instead of ordinary softmax of logits. Taking the metric learning idea into OOD detection. It is also the concurrent work of Generalized ODIN.
> </p>
> </details>


**[ACCV-2020]**
[Hyperparameter-free out-of-distribution detection using cosine similarity](https://openaccess.thecvf.com/content/ACCV2020/papers/Techapanurak_Hyperparameter-Free_Out-of-Distribution_Detection_Using_Cosine_Similarity_ACCV_2020_paper.pdf)
<br>
**Authors:** Engkarat Techapanurak, Masanori Suganuma, Takayuki Okatani
<br>
**Institution:** Tohoku University;  RIKEN Center for AIP
> <details>
> <summary> Using scaled cosine similarity between test sample features and class features to determine OOD samples. </summary>
> <p style="text-align:left">
> The first work employs softmax of scaled cosine similarity instead of ordinary softmax of logits. Taking the metric learning idea into OOD detection. It is also the concurrent work of Generalized ODIN.
> </p>
> </details>

**[ECCV-2020]**
[A boundary based out-of-distribution classifier for generalized zero-shot learning](https://arxiv.org/abs/2008.04872)
<br>
**Authors:** Xingyu Chen, Xuguang Lan, Fuchun Sun, Nanning Zheng
<br>
**Institution:** Xian Jiaotong University; Tsinghua University
> <details>
> <summary>Find the Boundaries for OOD Classification and finally compare the cosine distance between samples in the unit hypersphere.</summary>
> <p style="text-align:left">
> 
> </p>
> </details>


**[CVPR-2021]**
[Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces](https://openaccess.thecvf.com/content/CVPR2021/papers/Zaeemzadeh_Out-of-Distribution_Detection_Using_Union_of_1-Dimensional_Subspaces_CVPR_2021_paper.pdf)
<br>
**Authors:** Alireza Zaeemzadeh, Niccolò Bisagno, Zeno Sambugaro, Nicola Conci, Nazanin Rahnavard, Mubarak Shah
<br>
**Institution:** University of Central Florida; University of Trento
> <details>
> <summary> Calculating class membership probabilities in a union of 1-dimensional subspaces.</summary>
> <p style="text-align:left">
> The cosine similarities between the extracted feature and the class vectors are used to compute the class membership probabilities, using a Union of 1-dimensional subspaces. The 1-dimensional subspaces is spanned by the first singular vector of the feature vectors extracted from the training set. Feature vectors lie on a union of 1-dimensional subspaces helps OOD samples to be robustly detected.
> </p>
> </details>


**[ICML-2020]**
[Uncertainty estimation using a single deep deterministic neural network](https://arxiv.org/abs/2008.04872)
<br>
**Authors:** Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal
<br>
**Institution:** University of Oxford
> <details>
> <summary>Find the Boundaries for OOD Classification and finally compare the cosine distance between samples in the unit hypersphere.</summary>
> <p style="text-align:left">
> 
> </p>
> </details>


**[arXiv-2020]**
[Feature Space Singularity for Out-of-Distribution Detection](https://arxiv.org/abs/2011.14654)
<br>
**Authors:** Haiwen Huang, Zhihan Li, Lulu Wang, Sishuo Chen, Bin Dong, Xinyu Zhou
<br>
**Institution:** University of Oxford; Peking University; MEGVII Technology; etc.
> <details>
> <summary> Distance to Feature Space Singularity can measure OOD.</summary>
> <p style="text-align:left">
> It is observed that in feature spaces, OOD samples concentrate near a Feature Space Singularity (FSS) point, and the distance from a sample to FSS measures the degree of OOD. It can be exlained that moving speeds of features of other data depend on their similarity to the training data. During training, they use generated uniform noise or validation data as OOD.
> </p>
> </details>






<br>

[Back to Top](#top)

<br>

