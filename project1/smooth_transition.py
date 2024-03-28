from math import exp,sqrt,fabs
# todo: V 的单位是km/h
class smooth_transition(object):
    def __init__(self):
        self.parm_cognitive_load = 1/200
        self.t = 0
        self.max_torque_human = 10
        self.q1 = 0.5
        self.s1 = 1 # TODO: 确定这个值是什么
        self.delta_speed = 0
        self.n1 = 1
        self.n2 = 3
        self.alpha = 0

    def coorinator(self,v,human_torque,machine_toqure,position_expect,position_real):
        if self.alpha>=0.98:
            return 1,1,0
        v = v*3.6 # 转换为km/h
        human_torque_normalize = self.normalize(human_torque,self.max_torque_human)
        
        Q = self.evaluater_of_SA(v,human_torque_normalize)
        S = self.evaluater_of_HM_intention_similarity(human_torque,machine_toqure,position_expect,position_real)
        
        HM_state = 0
        self.alpha = 0
        if Q >= self.q1 and S < self.s1:
            HM_state = 1
            self.alpha = self.calcute_alpha(Q,1)
            delta_speed = 0
        elif Q < self.q1 and S < self.s1:
            HM_state = 2
            self.alpha = self.calcute_alpha(Q,1)
            delta_speed = self.n1 * self.parm_cognitive_load

        elif Q >= self.q1 and S >= self.s1:
            HM_state = 3
            self.alpha = self.calcute_alpha(Q,S)
            delta_speed = 0

        elif Q < self.q1 and S >= self.s1:
            HM_state = 4
            delta_speed = self.n2 * self.parm_cognitive_load * S

        return self.alpha,HM_state,delta_speed

    def normalize(self,x,x_max):
        return (x_max - x)/x_max

    def calcute_alpha(self,Q,S,eta=5,epsilon=10):
        alpha = 1/(1 + exp(eta - epsilon * Q * S))
        return alpha

    def evaluater_of_cognitive_load(self):
        self.t = self.t + 1
        return self.parm_cognitive_load * self.t

    def evaluater_of_muscle_states(self,v,human_torque):
        T_t = self.func2(v)
        muscle_states = self.func(human_torque,T_t)
        return muscle_states

    def evaluater_of_SA(self,v,human_torque,a1=1,a2=2,a3=3):
        cognitive_load = self.evaluater_of_cognitive_load()
        muscle_states  = self.evaluater_of_muscle_states(v,human_torque)
        Q = 1 - exp(- ((a1+muscle_states)**a2) * cognitive_load**a3  )
        return Q
    
    def evaluater_of_HM_intention_similarity(self,torque_machine,torque_human,position_expect,position_real):
        S = sqrt((fabs(torque_machine) - fabs(torque_human))**2) + 20*sqrt((position_expect - position_real)**2)
        return S 

    def func(self,T,T_t,a=12.5,b=0.5):
        return 1/( 1 + exp( -(a/T_t) * (T- b*T_t) )) + 1/( 1 + exp( (a/(1- T_t)) * (T- b*(T_t+1)) )) - 1

    def func2(self,v,v_max=60,c=3,d=0.5):
        return 1/( 1 + exp( -c * ( (v_max-v)/v_max -d) ))