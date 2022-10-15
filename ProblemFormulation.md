# Project 1

The current state, X, The cket is made of the  location of the rocket in relation to the target landing site, the current velocity of the rocket, and the rocket's angle.

State: $X = [d_x, d_y, v_x, v_y, \theta]^T$

The input, a, of the rocket consists of the acceleration, $\alpha$, which is in the direction the nose of the rocket is pointing, and the change in angle of the rocket.

Input: $a= [\alpha, \Delta \theta]$

Dynamics:

$$d_x(t+1) = d_x(t)+Vx(t)*\Delta t - \frac{1}{2}*\Delta t^2 \alpha sin(\theta)$$

$$d_y(t+1) = d_y(t)+Vy(t)*\Delta t - \frac{1}{2}*\Delta t^2 (\alpha*cos(\theta)-g)$$

$$v_x(t+1) = V_x(t)- \Delta t \alpha cos(\theta)$$
