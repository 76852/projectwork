
$\vec{x_i}=[x_1^i,x_2^i,\cdots,x_m^i]$             

$\vec{\beta}=[\beta_0,\beta_1,\cdots,\beta_m]$

#
$$
\begin{align*}
y_i \approx \beta_0 + \sum_{j=1}^m \beta_j \times x_j^i     
\end{align*}
$$


$\vec{x_i}=[1,x_1^i,x_2^i,\cdots,x_m^i]$

#
$$
\begin{align*}
y_i \approx \sum_{j=0}^m \beta_j \times x_j^i =\vec{\beta} \times \vec{x_i}
\end{align*}
$$

#
$$
\begin{align*}
\hat{\vec{\beta}} = \arg_{\vec{\beta}} \min L\left(D,\vec{\beta}\right) = 
\arg_{\vec{\beta}} \min \sum_{i=1}^n \left(\vec{\beta}\cdot\vec{x_i}-y_i\right)^2 
\end{align*}
$$

#
$$
\begin{align*}
L\left(D,\vec{\beta}\right) &= \lVert X\vec{\beta}-Y \rVert^2 \\
                            &= \left(X\vec{\beta}-Y\right)^T\left(X\vec{\beta}-Y\right) \\
                            &= Y^TY-Y^TX\vec{\beta}-\vec{\beta}^TX^TY+\vec{\beta}^TX^TX\vec{\beta} 
\end{align*}
$$

#
$$
\begin{align*}
\frac{\partial L \left(D,\vec{\beta}\right)}{\partial\vec{\beta}} &=
\frac{\partial \left(Y^TY-Y^TX\vec{\beta}-\vec{\beta}^TX^TY+\vec{\beta}^TX^TX\vec{\beta} \right)}
{\partial\vec{\beta}} \\
&= -2X^TY+2X^TX\vec{\beta} \\
\end{align*}
$$

#
$$
\begin{align*}
-2X^TY+2X^TX\vec{\beta} &= 0 \\
\Rightarrow X^TX\vec{\beta} &= X^TY \\
\Rightarrow \vec{\hat{\beta}} &= \left(X^TX\right)^{-1}X^TY
\end{align*}
$$

#
$$
L \left(D,\vec{\beta}\right)=\sum_i \left(y_i-\vec{\beta}\cdot\vec{x_i}\right)^2
$$

#
$$
\begin{align*}
H \left(D,\vec{\beta}\right) &= \prod_{i=1}^n P_r\left(y_i|\vec{x_i};\vec{\beta},\sigma \right)\\
&= \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp \left( -\frac{\left(y_i-\vec{\beta}\cdot\vec{x_i}\right)^2}{2\sigma^2} \right)
\end{align*}
$$

#
$$
\begin{align*}
I \left(D,\vec{\beta}\right) &= \log \prod_{i=1}^n P_r(y_i|\vec{x_i};\vec{\beta},\sigma)\\
&=\log \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp \left( -\frac{\left(y_i-\vec{\beta}\cdot\vec{x_i}\right)^2}{2\sigma^2} \right)\\
&=n\log \frac{1}{\sqrt{2\pi}\sigma} -\frac{1}{2\sigma^2} \sum_{i=1}^n \left(y_i-\vec{\beta}\cdot\vec{x_i}\right)^2
\end{align*}
$$

#
$$
\begin{align*}
\arg_{\vec{\beta}} \max I \left(D,\vec{\beta}\right) &= \arg_{\vec{\beta}} \max \left( n\log \frac{1}{\sqrt{2\pi}\sigma} -\frac{1}{2\sigma^2} \sum_{i=1}^n \left(y_i-\vec{\beta}\cdot\vec{x_i}\right)^2 \right)\\
&= \arg_{\vec{\beta}} \min \sum_{i=1}^n \left(y_i-\vec{\beta} \cdot \vec{x_i} \right)^2\\
&= \arg_{\vec{\beta}} \min L \left(D,\vec{\beta}\right)\\
&= \vec{\hat{\beta}}
\end{align*}
$$