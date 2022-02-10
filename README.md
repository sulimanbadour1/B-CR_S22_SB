# BCR_S22_SB
Behavioral and Cognitive Robotics _Exercises

the source code of one of the classic control problems, e.g. the pendulum.py file,
 what is encoded in the observation vector :
 ## Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |
 
 
 
 what is encoded in the action vector:

  ## Action Space
     The action is the torque applied to the pendulum.
      | Num | Action | Min  | Max |
      |-----|--------|------|-----|
      | 0   | Torque | -2.0 | 2.0 |
 
 the initial conditions are varied in the env.reset as follows:
  ## Starting State
        The starting state is a random angle in `[-pi, pi]` and a random angular velocity in `[-1,1]`.
 &
       An episode terminates after 200 steps. There's no other criteria for termination.
 
 the reward is calculated as :
 
      r = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
