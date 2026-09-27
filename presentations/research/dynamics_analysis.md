Yes. **A frozen-teacher JEPA can be modeled as learning a factorization of a fixed prediction problem: the encoder chooses the representation, while the predictor translates that representation into the teacher’s target space.**

Your intuition is right, but there is an important distinction:

> **The predictor should give the encoder freedom to choose its coordinates and organize information—not freedom to leave all useful computation to the predictor.**

The paper you linked, **SALT**, trains a reconstruction-based teacher, freezes it, and then trains a student encoder and predictor through masked latent prediction. Its student is therefore not required to reproduce the teacher representation directly at the encoder output. SALT’s actual prediction loss is \(L_1\); below, I use squared error where it makes the dynamics analytically tractable. ([arXiv][1])

I would analyze your question at three levels: **what representations the objective permits, how optimization chooses among them, and how to bias that choice toward useful encoder embeddings.**

## 1. What is the encoder actually learning?

Let \(C\) denote the visible context, \(Q\) the target queries or positions, and \(Y\) the frozen teacher’s target embeddings:

$$
Z=f_\theta(C),\qquad
\widehat Y=g_\phi(Z,Q),\qquad
Y=\operatorname{stopgrad}\!\left[\mathcal N(T(x)_M)\right].
$$

Here \(T\) is frozen, \(M\) selects the target tokens, and \(\mathcal N\) is a fixed target-normalization rule. \(Z\) can contain all your peak tokens, global tokens, and potentially pair representations.

For the squared-error model,

$$
\mathcal L(\theta,\phi)
=
\frac12\mathbb E\left[
\|g_\phi(f_\theta(C),Q)-Y\|^2
\right].
$$

### The encoder need not reproduce the teacher’s representation

For any invertible matrix \(A\), consider

$$
\widetilde f(C)=Af_\theta(C),
\qquad
\widetilde g(z,Q)=g_\phi(A^{-1}z,Q).
$$

Then

$$
\widetilde g(\widetilde f(C),Q)
=
g_\phi(f_\theta(C),Q).
$$

Consequently, the prediction loss is unchanged, provided the architecture can represent these transformations.

This is the encoder’s **coordinate freedom**: it can rotate, rescale, or mix its features, and the predictor can compensate. A sufficiently expressive predictor can also compensate for some nonlinear reparameterizations.

However:

$$
\boxed{
\text{Equivalent prediction performance}
\;\not\Rightarrow\;
\text{equally useful embedding geometry}.
}
$$

An invertible linear transformation preserves information and ideal, unconstrained linear decodability, but can substantially change cosine similarity and nearest-neighbor structure. An invertible nonlinear transformation can preserve information while making a linear probe much worse.

Thus “optimal embedding” needs a readout criterion: optimal for linear probing, cosine retrieval, an attentive readout, or something else. SALT’s frozen-backbone evaluation includes attentive probing; that is not the same claim as optimal raw cosine or linear geometry. 

### More fundamentally, the encoder learns what is predictable

Define the best prediction from the available input:

$$
m(C,Q)=\mathbb E[Y\mid C,Q].
$$

For a fixed encoder and an unrestricted predictor, the optimal predictor is

$$
g_f^*(Z,Q)=\mathbb E[Y\mid Z,Q].
$$

The resulting loss decomposes as

$$
\boxed{
\min_g\mathcal L(f,g)
=
\underbrace{
\frac12\mathbb E\|Y-m(C,Q)\|^2
}_{\text{irreducible prediction error}}
+
\underbrace{
\frac12\mathbb E
\left\|
m(C,Q)-\mathbb E[m(C,Q)\mid Z,Q]
\right\|^2
}_{\text{predictive information lost by the encoder}}.
}
$$

This follows directly from conditional expectation and the squared-error decomposition.

**The encoder is encouraged to preserve the information needed to predict the teacher’s conditional mean—not necessarily all the information contained in the teacher.** For coordinatewise \(L_1\), the corresponding optimal statistic is a conditional median rather than the mean.

For example, suppose the teacher target is

$$
Y=(S,N),
$$

where \(S\) is useful structure and \(N\) is unpredictable noise. When the context reveals \(S\) but not \(N\), a student representation \(Z=S\) can support the optimal prediction \((S,0)\). It can also have better downstream geometry than the teacher’s noisy representation.

This gives a mechanism by which a student can improve on its teacher without needing to copy it.

But there is a limitation: **the prediction objective supplies no reason to preserve a feature that does not improve prediction of its targets.** Merely making the encoder “freer” cannot resolve a mismatch between the pretraining task and the downstream task.

---

## 2. The training dynamics: the predictor controls the encoder’s learning signal

We can write the dynamics exactly for continuous-time gradient descent on a fixed collection of masked examples.

Stack their representations, predictions, and residuals:

$$
z=f_\theta(C),\qquad
\widehat y=g_\phi(z,Q),\qquad
r=\widehat y-y.
$$

Define

$$
J_f=\frac{\partial z}{\partial\theta},
\qquad
J_g=\frac{\partial\widehat y}{\partial z},
\qquad
J_\phi=\frac{\partial\widehat y}{\partial\phi}.
$$

Absorbing the batch-averaging factor into the learning rates, gradient flow gives

$$
\dot\theta
=
-\eta_eJ_f^\top J_g^\top r,
\qquad
\dot\phi
=
-\eta_pJ_\phi^\top r.
$$

Therefore, the representation itself evolves according to

$$
\boxed{
\dot z
=
-\eta_eJ_fJ_f^\top J_g^\top r.
}
$$

The key term is

$$
\boxed{
\nabla_z\mathcal L=J_g^\top r.
}
$$

**The encoder does not receive the teacher error directly. It receives the teacher error after it has been transformed by the predictor’s Jacobian.**

That gives the predictor two distinct roles.

**It adapts the coordinate system.** A change in encoder coordinates need not be penalized permanently, because the predictor can adapt.

**It selects and conditions the learning signal.** Directions that the predictor barely uses receive little prediction-driven gradient. Directions that it strongly amplifies can dominate encoder updates.

Locally, the predictor induces the metric

$$
G_z=J_g^\top J_g,
$$

because a representation perturbation \(\delta z\) changes predictions by approximately \(J_g\delta z\), giving

$$
\|\delta\widehat y\|^2
\approx
\delta z^\top G_z\delta z.
$$

A direction in the null space of \(J_g\) is “free” to first order—but **it also receives no learning signal from this loss**. Unconstrained dimensions do not automatically become useful dimensions.

### Loss improvement can come from either network

The residual evolves as

$$
\boxed{
\dot r
=
-
\left(
\underbrace{
\eta_eJ_gJ_fJ_f^\top J_g^\top
}_{\text{encoder contribution}}
+
\underbrace{
\eta_pJ_\phi J_\phi^\top
}_{\text{predictor contribution}}
\right)r.
}
$$

This is a parameter-block decomposition of the gradient-flow/kernel formulation. The Jacobians can change throughout training; no constant-kernel or infinite-width approximation is needed for this instantaneous identity. ([arXiv][2])

It exposes an important failure mode:

> **The prediction loss can improve mainly because the predictor is improving, while the encoder changes little or becomes no more useful for your downstream readout.**

For gradient flow without regularization,

$$
-\dot{\mathcal L}
=
\eta_e\|\nabla_\theta\mathcal L\|^2
+
\eta_p\|\nabla_\phi\mathcal L\|^2.
$$

These terms provide a useful diagnostic decomposition under that optimizer model. For AdamW, momentum, adaptive preconditioning, and weight decay must also be accounted for; raw gradient norms are not an exact loss-improvement attribution.

---

## 3. A solvable model of how encoder features emerge

To see the feature-learning dynamics explicitly, replace the encoder and predictor with linear maps:

$$
Z=WC,\qquad
\widehat Y=PWC.
$$

Assume centered data and whitened context,

$$
\mathbb E[CC^\top]=I,
$$

and define the fixed context–target cross-covariance

$$
R=\mathbb E[YC^\top].
$$

Then

$$
\mathcal L(W,P)
=
\frac12\|PW-R\|_F^2+\text{constant},
$$

with dynamics

$$
\boxed{
\dot W=\eta_eP^\top(R-PW),
\qquad
\dot P=\eta_p(R-PW)W^\top.
}
$$

This is a frozen-target instance of the deep-linear learning problem analyzed by Saxe and colleagues. ([arXiv][3])

Write

$$
R=U\operatorname{diag}(\sigma_i)V^\top.
$$

Under singular-vector-aligned initialization, each mode can be represented by an encoder amplitude \(a_i\) and predictor amplitude \(b_i\):

$$
\dot a_i
=
\eta_e b_i(\sigma_i-a_ib_i),
\qquad
\dot b_i
=
\eta_p a_i(\sigma_i-a_ib_i).
$$

With equal learning rates and balanced initialization \(a_i=b_i\), define \(u_i=a_ib_i\). Then

$$
\dot u_i=2\eta u_i(\sigma_i-u_i),
$$

whose solution, for \(\sigma_i>0\), is

$$
\boxed{
u_i(t)
=
\frac{\sigma_i}{
1+\left(\frac{\sigma_i}{u_i(0)}-1\right)e^{-2\eta\sigma_i t}
}.
}
$$

This yields several useful conclusions.

**Features emerge according to predictive cross-covariance strength.** Large-\(\sigma_i\) modes emerge earlier; weaker modes can remain poorly learned for much longer. These are context–target relationships, not simply high-variance teacher dimensions. The independent-mode solution is exact under the stated initialization assumptions, not for an arbitrary transformer. 

**A bottleneck selects predictive modes.** With encoder width \(d\), the best linear solution retains a rank-\(d\) approximation to \(R\). It need not resemble the teacher’s original coordinates.

**Optimization and regularization select a factorization.** The loss constrains \(PW\), not \(P\) and \(W\) separately. Learning rates, initialization, normalization, and weight penalties influence how the solution is divided between the two networks.

For example, in this linear model, adding

$$
\frac{\lambda_e}{2}\|W\|_F^2+
\frac{\lambda_p}{2}\|P\|_F^2
$$

shrinks the optimal retained product amplitudes to

$$
u_i^*=
\left(\sigma_i-\sqrt{\lambda_e\lambda_p}\right)_+.
$$

This is an \(L_2\)-regularized linear-model result, **not a formula for plugging in AdamW weight-decay values**. Its conceptual implication is that regularization can suppress weak predictive modes rather than merely choose a convenient coordinate system.

So the aim should not be unrestricted freedom. It should be **well-conditioned learning of the useful predictive modes**.

---

## 4. The cleanest mathematical formulation of “more freedom”: optimize out the predictor

Your idea has a precise formulation:

$$
\boxed{
\mathcal L_{\mathrm{profile}}(\theta)
=
\min_\phi
\left[
\mathcal L(\theta,\phi)+\lambda R(\phi)
\right].
}
$$

Rather than judging an encoder using a possibly poorly adapted predictor, judge it using the best predictor available within a chosen predictor class.

### Why this can help

Suppose the encoder changes its internal basis but preserves the useful information. A lagging predictor may temporarily produce worse predictions and push the encoder back toward its old basis.

A predictor that adapts sufficiently quickly can absorb the basis change instead.

At an exact, differentiable inner optimum, the envelope theorem gives

$$
\nabla_\theta\mathcal L_{\mathrm{profile}}
=
\left.
\nabla_\theta\mathcal L(\theta,\phi)
\right|_{\phi=\phi^*(\theta)}.
$$

There is also a useful local curvature interpretation. With a nonsingular positive-definite predictor Hessian,

$$
\boxed{
H_{\mathrm{profile}}
=
H_{\theta\theta}
-
H_{\theta\phi}
H_{\phi\phi}^{-1}
H_{\phi\theta}.
}
$$

The subtracted term describes how predictor adaptation relaxes the encoder’s local constraints. Encoder motions that can be compensated by predictor motions become less costly.

**This is a mathematical expression of the freedom you are describing.**

### The linear-head case makes it especially clear

Let \(Z\in\mathbb R^{N\times d}\) contain encoder embeddings and let \(Y\) contain corresponding fixed target vectors. With an unrestricted linear decoder,

$$
\min_D\frac12\|ZD-Y\|_F^2
=
\frac12\|(I-\Pi_Z)Y\|_F^2,
$$

where \(\Pi_Z=ZZ^\dagger\) projects onto the column space of \(Z\).

The objective depends on **which target information the representation spans**, not on its particular basis.

For numerical stability, ridge regression gives

$$
D^*=(Z^\top Z+\lambda I)^{-1}Z^\top Y.
$$

Ridge introduces scale preferences, so it no longer has full invariance to arbitrary invertible coordinate changes. Nevertheless, it still avoids directly prescribing the encoder’s coordinates.

A practical caution: solving an unrestricted head on a batch with \(d\ge N\) can permit interpolation and provide little useful representation-learning signal. Use sufficiently many training examples, regularization, and held-out evaluation.

### But an optimal, unrestricted predictor can remove the incentive to learn

There is a counterexample worth keeping in mind.

Suppose \(f_\theta(C)\) is an injective but unstructured encoding of the input. A universal predictor can invert that encoding and perform the entire prediction task itself.

Then the best prediction loss may already be achievable without making the encoder representation more accessible or semantically organized.

Therefore,

$$
\boxed{
\text{Better predictor adaptation}
\neq
\text{unlimited predictor capacity}.
}
$$

I would distinguish **predictor speed** from **predictor computational responsibility**.

A modestly faster predictor can track representation changes. A much deeper predictor can instead take over feature extraction. Related non-contrastive SSL analyses also find that predictor dynamics can help feature learning, but that excessively fast predictor optimization can be harmful; those results concern different teacher dynamics and should not be transferred to frozen-teacher JEPA as a theorem. ([arXiv][4])

For an initial experiment, I would test predictor learning-rate ratios such as \(1\times,2\times,4\times\), separately from changing predictor depth.

---

## 5. How I would encourage useful freedom in your encoder

### A. Make the predictor adaptable without making it responsible for everything

The architectural principle I would test is:

$$
\text{encoder: reusable structure extraction}
\quad\longrightarrow\quad
\text{predictor: query-dependent decoding and coordinate adaptation}.
$$

For a token-based encoder, that suggests comparing your existing predictor with a shallower query-conditioned predictor while retaining enough width to avoid an unnecessary information bottleneck.

Width and depth should be ablated separately. Widening an interface can preserve more feature directions; adding layers can increase how much computation the predictor performs.

This is not an argument that a linear predictor is universally sufficient. SALT itself uses 12-layer, width-384 predictors of roughly 22 million parameters. The paper does not establish an optimal predictor-to-encoder capacity ratio for your setting. 

### B. Add a weak “easy to read” objective on the representation you actually export

This is the intervention I would prioritize when the masked-prediction loss improves but downstream encoder quality does not.

Keep the main predictor for difficult masked prediction, and add a simple readout:

$$
\widehat Y_{\mathrm{aux}}
=
D\,h_\theta(C),
$$

where \(h_\theta(C)\) is the actual representation used downstream: a global embedding, pooled peak embeddings, or a particular pair-derived representation.

Train with

$$
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{masked}}
+
\beta\mathcal L_{\mathrm{simple\ readout}}.
}
$$

The auxiliary targets could be an appropriate pooled frozen-teacher representation or another fixed target derived from its tokens.

The distinction is important:

$$
h_\theta(C)=Y_{\mathrm{aux}}
\quad\text{is not required;}
\qquad
Dh_\theta(C)\approx Y_{\mathrm{aux}}
\quad\text{is required}.
$$

The encoder retains coordinate freedom, but useful target information must be accessible through a simple map.

For your setup, **a predictor consuming all peak tokens does not by itself require a useful CLS embedding**. Likewise, good prediction through one representation stream does not guarantee that another exported stream has good retrieval geometry.

Keep this auxiliary objective weak initially and inspect its **encoder-gradient contribution**, not only its scalar loss. When a global matching branch already dominates training, rebalancing it is preferable to adding another dominant global objective.

This is a proposed modification, not a demonstrated SALT result.

### C. Regularize representation statistics without copying teacher geometry

A teacher-matching loss on the encoder output fixes a particular representation. A statistical regularizer can instead discourage degeneracy while leaving much more representational freedom.

For example, with \(C_Z=\operatorname{Cov}(Z)\), a VICReg-style penalty is

$$
\mathcal R(Z)
=
\lambda_v\sum_j
\left[
\gamma-\sqrt{(C_Z)_{jj}+\epsilon}
\right]_+
+
\lambda_c\sum_{i\ne j}(C_Z)_{ij}^2.
$$

The variance term discourages constant dimensions; the covariance term discourages redundancy. These mechanisms come from VICReg, rather than from the frozen-teacher objective itself. ([arXiv][5])

However, **high rank is not the same as useful information**. A regularizer can preserve noise just as readily as semantics. I would use it to address an observed degeneracy, not assume that stronger whitening must improve representations.

Also, measure covariance across examples in the representation you actually deploy. Pooling every peak token into one covariance estimate can conceal poor spectrum-level representations behind within-spectrum token variation.

LayerNorm alone does not resolve this: identical vectors across examples can all have perfectly normal per-example norms.

### D. Change which teacher directions dominate, rather than only changing encoder flexibility

The target metric determines which prediction errors matter.

For a weighted squared loss,

$$
\mathcal L
=
\frac12\mathbb E[r^\top Mr],
$$

the local encoder-space metric becomes

$$
G_Z=J_g^\top M J_g.
$$

Thus teacher scaling and predictor conditioning act together.

One controlled experiment is fixed, shrinkage-based target normalization:

$$
\widetilde Y
=
(\Sigma_T+\epsilon I)^{-\alpha/2}(Y-\mu_T),
\qquad 0\le\alpha\le1.
$$

Here \(\alpha=0\) leaves relative target scales unchanged, while \(\alpha=1\) approximately whitens them.

This changes which predictive modes training prioritizes; it is not merely a cosmetic normalization. Full whitening can amplify low-variance noise, so I would compare mild normalization with the baseline and keep all estimated target statistics fixed.

Most importantly, **normalizing teacher targets does not guarantee noncollapsed student embeddings**. It changes the prediction problem; it does not directly constrain the student’s representation covariance.

### E. Ensure the prediction task requires the information you want

The conditional-mean analysis tells us where to look:

$$
\mathbb E[Y\mid C,Q]
$$

must depend on useful properties of \(C\).

For mass spectra, this deserves particular attention because a target peak’s \(m/z\) is content, not merely a fixed spatial address. Supplying it as a query changes what the predictor must infer. That is not automatically wrong, but it can make part of the task solvable without much contextual understanding.

I would train a **query-and-metadata-only baseline** and compare it with the full model. This measures how much prediction actually benefits from the encoder.

Mask design should then encourage the relationships you care about, while leaving enough context to make them inferable. Making masks harder is not automatically better: when the desired target becomes unpredictable, the optimal prediction simply moves toward a conditional mean or median.

---

## 6. Frozen targets remove one collapse mechanism—not every bad solution

A frozen nonconstant teacher prevents the student and teacher from jointly drifting toward the same trivial constant target.

But it does not establish that every encoder dimension remains informative.

From the decomposition above, a constant encoder can even be optimal for squared error when

$$
\mathbb E[Y\mid C,Q]=\mathbb E[Y\mid Q].
$$

In that case, context adds no conditional-mean prediction value, even though the teacher targets themselves may vary substantially.

Less extreme possibilities include partial rank collapse, retaining only easy features, poor global pooling, and a predictor that handles most of the useful computation.

SALT reports strong correlations between prediction loss and downstream probe accuracy in its experiments. That is valuable evidence for its configuration, but it is not a universal monotonicity guarantee for another modality, masking scheme, or predictor. 

---

## 7. The experiments I would run first

I would use the same frozen teacher and a small, controlled student-training budget:

| Experiment                                            | Main question                                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| Current encoder and predictor                         | What is the baseline relationship between prediction loss and encoder quality? |
| Same architecture, modestly faster predictor          | Is predictor lag restricting useful encoder changes?                           |
| Shallower predictor, with width tested separately     | Is too much useful computation occurring after the encoder?                    |
| Baseline plus weak simple-readout auxiliary           | Is useful information insufficiently accessible in the exported embedding?     |
| Baseline plus mild variance/covariance regularization | Is representation degeneracy contributing to the problem?                      |
| Baseline with fixed target rescaling                  | Are a few teacher directions dominating optimization?                          |

Compare at matched compute as well as matched steps, because extra predictor updates are not free. When changing target normalization, retain a common held-out prediction metric so that differently scaled training losses are not compared directly.

At checkpoints, I would examine three things together.

**Encoder utility.** Evaluate the actual exported representation using the same linear probe, lightweight nonlinear probe, and retrieval protocol. This distinguishes absent information from information that is present but difficult to access.

**Predictor catch-up.** Freeze the encoder and fit a fresh, standardized predictor on training embeddings, then evaluate on held-out examples. A substantial improvement over the online predictor suggests predictor underoptimization. A large advantage for a deep readout over a shallow one suggests that substantial computation is still needed after encoding, although that alone does not prove poor representation quality.

**Representation health.** Track covariance spectra, effective rank, feature norms, and across-example variation. Interpret norm changes cautiously: encoder scaling can be compensated by predictor scaling without changing information.

## My recommendation

For your frozen-teacher setup, I would **not start by making the predictor much larger**.

I would first separate predictor speed from predictor capacity, then test a **weak simple-readout objective on the encoder representation actually used for your probes or retrieval**. That directly targets the distinction between “the full encoder–predictor system can solve the task” and “the encoder has learned useful embeddings.”

The resulting design would be

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{masked\ latent\ prediction}}
+
\beta\mathcal L_{\mathrm{simple\ readout}}
+
\gamma\mathcal R_{\mathrm{representation}},
}
$$

with the last term added only when the diagnostics justify it.

**The desired freedom is freedom to choose a useful representation—not freedom to make useful information arbitrarily difficult to decode.** A predictor that adapts to coordinates, a constrained readout that demands accessibility, and targets that require the right information together provide a much clearer route toward that goal.

[1]: https://arxiv.org/html/2509.24317v1 "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers"
[2]: https://arxiv.org/abs/1806.07572 "[1806.07572] Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
[3]: https://arxiv.org/abs/1312.6120 "[1312.6120] Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
[4]: https://arxiv.org/pdf/2102.06810 "Understanding Self-Supervised Learning Dynamics without Contrastive Pairs"
[5]: https://arxiv.org/abs/2105.04906 "[2105.04906] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
