# Project 1
### State: 

The current state, X, The cket is made of the  location of the rocket in relation to the target landing site, the current velocity of the rocket, and the rocket's angle.
 
$$X = [d_x, d_y, v_x, v_y, \theta]^T$$


### Input: 
The input, a, of the rocket consists of the acceleration, $\alpha$, which is in the direction the nose of the rocket is pointing, and the change in angle of the rocket.

$$a= [\alpha, \Delta \theta]$$

### Dynamics:

$$d_x(t+1) = d_x(t)+Vx(t)*\Delta t - \frac{1}{2}*\Delta t^2 (\alpha sin(\theta) + C_d \rho v_x(t)^2 )$$

$$d_y(t+1) = d_y(t)+Vy(t)*\Delta t + \frac{1}{2}*\Delta t^2 (\alpha*cos(\theta)-g + C_d \rho v_y(t)^2)$$

$$v_x(t+1) = v_x(t)- \Delta t (\alpha sin(\theta) + C_d \rho v_x(t)^2) $$

$$v_y(t+1) = v_y(t)+ \Delta t (\alpha cos(\theta)+ C_d \rho v_y(t)^2 - g)$$

$$\theta (t+1) = \theta (t) + \Delta \theta$$

The model of drag used in this problem is vastly simplified and will be soley a function of the airspeed, the air desity and a constant, where $C_d$ is the coeffifient of drag and $rho$ is the air density as a function of $d_y$. Likewise, the model used for air density has been simplified down to:

$$\rho  =\rho _{b}\exp \left[{\frac {-g_{0}M\left(h-h_{b}\right)}{R^{*}T_{b}}}\right] =1.2250 *\exp{\left[{\frac {-9.81 * .0289644\left(d_y\right)}{8.3145^{*}(288.15)}}\right]} = 1.2250 *\exp{[-1.186 * 10^{-4}* d_y]}$$

### Constraints:

$$ -\alpha_{max} \leq \alpha \leq \alpha_{max} $$

$$ -\Delta\theta_{max} ^{\circ} \leq \Delta\theta \leq \Delta\theta_{max}^{\circ} $$

$$ d_y \geq 0 $$

$$ t \leq T_{max} $$

### Target: 
at $t = T_{max},\ X_T= [0, 0, 0 ,0 ,0]^T $

### Controller:

We will use a contoller $\pi_\phi$, where $\phi$ are the design variables and $\pi$ is a neural network which takes the state X as the input and returns the output $\alpha$. To find the optimized pararamters we will define the error, e, as:

$$ e = \|| W \cdot (X(T_{max}) - X_T) \||$$

Where W is a vector containing the weights of each state. 

Therefore, the objective is:

$$ min_\phi \ e $$
