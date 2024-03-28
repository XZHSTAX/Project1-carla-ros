from agents.navigation.behavior_agent import BehaviorAgent
from project1.util.carla_util import CarlaSyncMode, draw_image_np,carla_img_to_array
import random
import carla
import pygame
import copy
import numpy as np
import math
from pygame.locals import K_p
from configparser import ConfigParser
from pygame.locals import *
from project1.MPC import MPCPlusPID
from project1.pure_pursuit import PurePursuitPlusPID
import rclpy
from rclpy.node import Node
from ros_g29_force_feedback.msg import ForceFeedback
from std_msgs.msg import Float64
import threading
from threading import Thread
from scipy.signal import butter, lfilter
from project1.smooth_transition import smooth_transition
from collections import deque
import csv
import time
color_bar = [(0,   255,  0 , 0),   # Green  LEFT
             (255, 0,    0 , 0),   # Red    RIGHT
             (0,   0,   0  , 0),   # Black  STRAIGHT
             (255, 255, 255, 0),   # White  LANEFOLLOW
             (0,   0,   255, 0),   # Blue   CHANGELANELEFT
             (255, 255,  0 , 0),   # Yellow CHANGELANERIGHT
             ]
FPS = 37
plan_fre = 3    # 实际上这里的频率指的是，实际程序运行多少次，mpc规划控制运行一次
mpc_sample_time = 1./FPS * plan_fre

main_image_shape = (1280, 720)
driver_real_view = [[0.1, -0.25 ,1.25],[0,0,0]]    # 正常开车视角
driver_sim_view1 = [[1.25, 0,    1.25],[0,0,0]]    # 模拟视角，前方视角
driver_sim_view2 = [[-10, 0,     2.8], [-10,0,0]]  # 模拟视角，后看视角
driver_sim_view3 = [[-5.5, 0,     2.8], [-10,0,0]]  # 模拟视角，后看视角

driver_view = driver_sim_view3

use_HSC_mode = 0
ct_point = carla.Location(-82.68668,4.825,0)

class MainNode(Node):
    def __init__(self,world):
        super(MainNode, self).__init__("MainNode")
        self.publisher = self.create_publisher(ForceFeedback,"/ff_target", 10)
        self.subscriber = self.create_subscription(Float64,"external_torque",self.listener_callback,10)
        self.subscriber_machine_torque = self.create_subscription(Float64,"machine_torque",self.listener_callback2,10)
        self.world = world
        self.clock = pygame.time.Clock()
        self.controller_human = Control_with_G29()
        self.controller_machine = MPCPlusPID()
        self.controller_machine2 = PurePursuitPlusPID()
        self.use_controller = 2

        self.HSC = smooth_transition()
        self.prepare_wold()
        
        pygame.init()
        self.display = pygame.display.set_mode(
            main_image_shape,
            pygame.HWSURFACE | pygame.DOUBLEBUF)    # 创建游戏窗口
        self.font = pygame.font.SysFont('ubuntumono', 14) # 设置字体
        self.font_speed = pygame.font.SysFont('ubuntumono', 100) # 设置字体

        self.plan_count = 0
        self.reach_ct_point = 0
        self.shared_control_transform = 0
        self.external_torque = 0
        fs = 1/0.002
        cutoff_freq = 15
        self.LowPassFilter_for_ex_torque =  LowPassFilter(cutoff_freq,fs)
        self.desired_speed = 15
        self.alpha = 0
        self.HM_state = 0
        self.HSC_color_bar = [(255,255,255),(255, 247, 138),(255, 227, 130),(255, 196, 126),(255, 173, 132)]
        self.delta_speed = 0
        self.recorder = {"X":[],
                         "Y":[],
                         "Yaw":[],
                         "Speed":[],
                         "Torque_machine":[],
                         "Torque_human":[],
                         "Steering_angle_machine":[],
                         "Steering_angle_real":[],
                         "HM_state":[],
                         "Time":[],
                         "Alpha":[],
                         "Manual_Auto_HSC":[],
                         }
        self.old_speed = deque(maxlen=50)
        self.steering_angle_machine = 0
        self.Manual_Auto_HSC = 0
        self.machine_torque = 0.0
        self.record_time = time.time()
        self.common_takeover = 0


    def prepare_wold(self):
        self.actor_list = []
        m = self.world.get_map()

        blueprint_library = self.world.get_blueprint_library()

        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))  # 选择车辆
        veh_bp.set_attribute('color','64,81,181')                            # 上色
        veh_bp.set_attribute('role_name', 'hero')
        self.vehicle = self.world.spawn_actor(                                         # 生成车辆
            veh_bp,
            m.get_spawn_points()[56])                                        # 车辆位置选择为生成点[90]
        self.actor_list.append(self.vehicle)

        # 相机输出的画面大小
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x',str(main_image_shape[0]))
        camera_bp.set_attribute('image_size_y',str(main_image_shape[1]))

        # visualization cam (no functionality)
        camera_rgb = self.world.spawn_actor(
                                    camera_bp,
                                    carla.Transform(carla.Location(*driver_view[0]), carla.Rotation(*driver_view[1])),
                                    attach_to=self.vehicle)
        
        self.actor_list.append(camera_rgb)
        self.sensors = [camera_rgb]

        self.destination = m.get_spawn_points()[294].location
        self.route,self.agent = autopilot(self.vehicle,self.destination)
        draw_waypoint(self.route,self.world)
        if self.use_controller == 1:
            self.traj = waypoint2traj(self.route,self.vehicle,transform2vehicle=0,contain_yaw=1)
        elif self.use_controller == 2:
            self.traj = waypoint2traj(self.route,self.vehicle)

    def main_loop(self):
        with CarlaSyncMode(self.world, *self.sensors,fps=FPS) as sync_mode: 
            while rclpy.ok():
                self.clock.tick() 
                tick_response = sync_mode.tick(timeout=2.0)
                speed = np.linalg.norm( carla_vec_to_np_array(self.vehicle.get_velocity()))
                state = [self.vehicle.get_transform().location.x, 
                         self.vehicle.get_transform().location.y, 
                         self.vehicle.get_transform().rotation.yaw,
                         speed,
                         0.5*(self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)+self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))]

                if self.controller_human.parse_events(self.agent,self.destination): # 如果按下esc键，则直接退出
                    return
                
                if self.controller_human._autopilot_enabled: # 如果使用自动驾驶
                    if self.use_controller == 1:             # 如果使用控制器1
                        if self.plan_count%plan_fre == 0:    # 如果到了使用次数
                            traj_object = waypoint2traj(self.route,self.vehicle,transform2vehicle=1,contain_yaw=1)
                            throttle, self.steering_angle_machine = self.controller_machine.get_control(self.traj,traj_object, desired_speed=self.desired_speed, dt=mpc_sample_time,state = state)
                        self.plan_count = self.plan_count + 1
                    elif self.use_controller == 2:
                        self.traj = waypoint2traj(self.route,self.vehicle)
                        throttle, self.steering_angle_machine = self.controller_machine2.get_control(self.traj,speed,desired_speed=self.desired_speed,dt=1./FPS)
                    self.Manual_Auto_HSC = 1


                    send_control(self.vehicle,throttle,self.steering_angle_machine,0)
                    out_msg = ForceFeedback()
                    out_msg.position = np.clip(self.steering_angle_machine, -1.0, 1.0)
                    out_msg.torque = 0.8
                    self.publisher.publish(out_msg)

                elif self.shared_control_transform:
                    # 如果开启平滑过度模式
                    self.alpha,self.HM_state,self.delta_speed = self.HSC.coorinator(speed,
                                                                    self.external_torque,
                                                                    self.machine_torque,
                                                                    self.steering_angle_machine,
                                                                    self.controller_human._control.steer)
                    self.desired_speed = self.desired_speed - self.delta_speed
                    if self.use_controller == 1:
                        if self.plan_count%plan_fre == 0:
                            traj_object = waypoint2traj(self.route,self.vehicle,transform2vehicle=1,contain_yaw=1)
                            throttle, self.steering_angle_machine = self.controller_machine.get_control(self.traj,
                                                                                traj_object, 
                                                                                desired_speed=self.desired_speed, 
                                                                                dt=mpc_sample_time,
                                                                                state = state)
                        self.plan_count = self.plan_count + 1
                    elif self.use_controller == 2:
                        self.traj = waypoint2traj(self.route,self.vehicle)
                        throttle, self.steering_angle_machine = self.controller_machine2.get_control(self.traj,speed,desired_speed=self.desired_speed,dt=1./FPS)
                    
                    send_control(self.vehicle,throttle,
                                        self.alpha * self.controller_human._control.steer + (1- self.alpha)*self.steering_angle_machine,
                                        self.controller_human._control.brake,
                                        self.controller_human._control.hand_brake,
                                        self.controller_human._control.reverse)
                    out_msg = ForceFeedback()
                    out_msg.position = np.clip(self.alpha * self.controller_human._control.steer + (1- self.alpha)*self.steering_angle_machine, -1.0, 1.0)
                    out_msg.torque = 1.0 - self.alpha
                    self.publisher.publish(out_msg)
                    if self.alpha >= 0.98 :
                        self.shared_control_transform = 0 
                    self.Manual_Auto_HSC = 2                    
                
                elif self.common_takeover:
                    self.traj = waypoint2traj(self.route,self.vehicle)
                    throttle, self.steering_angle_machine = self.controller_machine2.get_control(self.traj,speed,desired_speed=self.desired_speed,dt=1./FPS)
                    if self.controller_human.human_intervent:
                        send_control(self.vehicle,self.controller_human._control.throttle,
                                            self.controller_human._control.steer,
                                            self.controller_human._control.brake,
                                            self.controller_human._control.hand_brake,
                                            self.controller_human._control.reverse,throttle_limit = 0.5)
                        out_msg = ForceFeedback()
                        out_msg.position = 0.0
                        out_msg.torque = 0.3
                        self.publisher.publish(out_msg)
                    else:
                        send_control(self.vehicle,throttle,self.steering_angle_machine,0)


                else:# 手动控制
                    send_control(self.vehicle,self.controller_human._control.throttle,
                                        self.controller_human._control.steer,
                                        self.controller_human._control.brake,
                                        self.controller_human._control.hand_brake,
                                        self.controller_human._control.reverse,throttle_limit = 0.5)
                    out_msg = ForceFeedback()
                    out_msg.position = 0.0
                    out_msg.torque = 0.3
                    self.publisher.publish(out_msg)
                    self.Manual_Auto_HSC = 0
                
                snapshot, image_rgb = tick_response
                image_rgb = copy.copy(carla_img_to_array(image_rgb))
                draw_image_np(self.display, image_rgb)                   # 在Pygame显示中绘制图像
                
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
                texts = ["FPS (real):                % 3.0f "%int(self.clock.get_fps()),
                         "FPS (simulated):           % 3.0f "%fps,
                         "speed (m/s):               % 3.0f" %speed,
                         'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y)),
                         'Yaw:                       % 3.0f' %self.vehicle.get_transform().rotation.yaw,
                         'external torque(N·m):      % 3.1f'%self.external_torque,
                         'machine torque(N·m):       % 3.1f'%self.machine_torque,
                         'Alpha:                     % 3.1f'%self.alpha,
                         'delta speed:               % 3.1f'%self.delta_speed,
                         'desired speed:             % 2.1f'%self.desired_speed,
                        ]                
                self.display_info(self.display,texts,self.font,self.vehicle,speed,self.font_speed)
                                
                # 如果到达接管触发点，则结束完全自动驾驶，开启
                if not self.reach_ct_point:
                    if reach_destination(self.vehicle,ct_point,radius=2):
                        if use_HSC_mode:
                            self.controller_human._autopilot_enabled = 0
                            self.reach_ct_point = 1
                            self.shared_control_transform = 1
                        else:
                            self.reach_ct_point = 1
                            self.common_takeover = 1
                            self.controller_human._autopilot_enabled = 0
                if reach_destination(self.vehicle,self.destination):
                    self.controller_human._autopilot_enabled = 0
                
                if self.controller_human.start_to_record:
                    time_now = time.time()
                    time_passed = time_now - self.record_time
                    self.record_time = time_now
                    self.record_data([state[0], # X
                                    state[1], # Y
                                    state[2], # Yaw
                                    state[3], # Speed
                                    self.machine_torque,                   # Torque_machine
                                    self.external_torque,                  # Torque_human
                                    self.steering_angle_machine,           # Steering_angle_machine
                                    self.controller_human._control.steer,  # Steering_angle_real
                                    self.HM_state,                         # HM_state
                                    time_passed,                                   # Time
                                    self.alpha,
                                    self.Manual_Auto_HSC]) 
                

    def write_text(self,text,font,display,text_color=(255,0,0),text_location=None):
        text_surface = font.render(text, True, text_color)
        
        if not text_location:
            text_width = text_surface.get_width()
            text_height = text_surface.get_height()
            text_x = (main_image_shape[0] - text_width) // 2
            text_y = (main_image_shape[1] - text_height) // 2
        else:
            text_x, text_y = text_location

        display.blit( text_surface, (text_x, 0))

    def draw_progress_bar(self,display,frame_location,frame_size,progress_location,progress_size,frame_color=(255,255,255),progress_color=(255,255,255)):
        rect_border = pygame.Rect(frame_location, frame_size)
        pygame.draw.rect(display, frame_color, rect_border, 1)

        rect = pygame.Rect(progress_location, progress_size)
        pygame.draw.rect(display, progress_color, rect)        

    def display_info(self,display,texts,font,vehicle,speed,font_speed,dy=18):
        info_surface = pygame.Surface((220, main_image_shape[1]/3))
        info_surface.set_alpha(100)
        display.blit(info_surface, (0, 0))    
        
        for it,t in enumerate(texts):
            display.blit(
                font.render(t, True, (255,255,255)), (5, 20+dy*it))
        v_offset =  20+dy*it + dy

        throttle_rate = vehicle.get_control().throttle
        display.blit( font.render("Throttle:", True, (255,255,255)), (5, v_offset))
        # throttle_rate = np.clip(throttle, 0.0, 1.0)
        
        bar_width = 106
        bar_h_offset = 100
        self.draw_progress_bar(display,
                               (bar_h_offset, v_offset+6),
                               (bar_width, 6),
                               (bar_h_offset, v_offset+6),
                               (throttle_rate * bar_width, 6))
        v_offset = v_offset + dy
        #---------------------------------------------------------------------------------#
        display.blit( font.render("Steer:", True, (255,255,255)), (5, v_offset))
        # steer_rate = np.clip(steer, -1.0, 1.0)
        G29_steer_rate = self.controller_human._control.steer
        self.draw_progress_bar(display,
                               ((bar_h_offset+(G29_steer_rate+1)/2*bar_width), v_offset+6),
                               (6, 6),
                               ((bar_h_offset+(G29_steer_rate+1)/2*bar_width), v_offset+6),
                               (5, 5),
                               frame_color=(0,0,0),
                               progress_color=(255,0,0))        
        
        steer_rate = vehicle.get_control().steer
        self.draw_progress_bar(display,
                               (bar_h_offset, v_offset+6),
                               (bar_width, 6),
                               ((bar_h_offset+(steer_rate+1)/2*bar_width), v_offset+6),
                               (6, 6))
        v_offset = v_offset + dy
        #----------------------------------------------------------------------------#
        display.blit( font.render("REC:", True, (255,0,0)), (5, v_offset))
        steer_rate = vehicle.get_control().steer
        self.draw_progress_bar(display,
                               (bar_h_offset+bar_width, v_offset+6),
                               (6, 6),
                               ((bar_h_offset+bar_width), v_offset+6),
                               (6, 6),
                               progress_color = (255,0,0) if self.controller_human.start_to_record else (0,0,0))
        v_offset = v_offset + dy


        if all(speed > x for x in self.old_speed):
            Speed_color = (255, 105, 105)
        elif all(speed < x for x in self.old_speed):
            Speed_color = (0, 0, 255)
        else:
            Speed_color = (255,255,255)
        self.old_speed.append(speed)
        speed_str = "%4.1fkm/h" %(speed*3.6)
        display.blit( font_speed.render(speed_str, True, Speed_color), (main_image_shape[0]-400,main_image_shape[1]-90) )
        
        
        if self.controller_human._autopilot_enabled:
            autopilot_on_str = "Auto"
            self.write_text(autopilot_on_str,font_speed,display)
        elif self.shared_control_transform:
            HSC_str = "HSC %s"%self.HM_state
            self.write_text(HSC_str,font_speed,display,self.HSC_color_bar[self.HM_state])           
            bar_width = 800
            bar_h_offset = 500
            bar_location_x = (main_image_shape[0] - bar_width)// 2
            bar_height = 25
            self.draw_progress_bar(display,
                                (bar_location_x, 100),
                                (bar_width, bar_height),
                                (bar_location_x, 100),
                                (self.alpha * bar_width, bar_height),
                                progress_color=self.HSC_color_bar[self.HM_state]) 
        else:
            autopilot_off_str = "Auto OFF"
            self.write_text(autopilot_off_str,font_speed,display)

        pygame.display.flip() # 将绘制的图像显示在屏幕上  

    def listener_callback(self,msg):
        external_torque_filted = self.LowPassFilter_for_ex_torque.filter(msg.data)
        self.external_torque = external_torque_filted

    def listener_callback2(self,msg):
        self.machine_torque = msg.data

    def spin(self):
            rclpy.spin(self, self.executor)
            
    def record_data(self,data_list):
        self.recorder["X"].append(data_list[0])
        self.recorder["Y"].append(data_list[1])
        self.recorder["Yaw"].append(data_list[2])
        self.recorder["Speed"].append(data_list[3])
        self.recorder["Torque_machine"].append(data_list[4])
        self.recorder["Torque_human"].append(data_list[5])
        self.recorder["Steering_angle_machine"].append(data_list[6])
        self.recorder["Steering_angle_real"].append(data_list[7])
        self.recorder["HM_state"].append(data_list[8])
        self.recorder["Time"].append(data_list[9])
        self.recorder["Alpha"].append(data_list[10])
        self.recorder["Manual_Auto_HSC"].append(data_list[11])


class Control_with_G29(object):
    '''
    设计一个类，用于检测方向盘的输入，并且转换为对应的控制值，存储在成员变量中
    '''
    def __init__(self):
        self._control = carla.VehicleControl()
        # self.autopilot_on = False
        self._control.steer = 0.0
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._control.hand_brake = False
        self._control.reverse = False

        self._autopilot_enabled = False
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()    # rclpy.init(args=args)
 
        self._parser = ConfigParser()
        self._parser.read('src/project1/project1/wheel_config.ini')
        self._steer_idx     = int( self._parser.get( 'G29 Racing Wheel', 'steering_wheel' ) )
        self._throttle_idx  = int( self._parser.get( 'G29 Racing Wheel', 'throttle' ) )
        self._brake_idx     = int( self._parser.get( 'G29 Racing Wheel', 'brake' ) )
        self._reverse_idx   = int( self._parser.get( 'G29 Racing Wheel', 'reverse' ) )
        self._handbrake_idx = int( self._parser.get( 'G29 Racing Wheel', 'handbrake' ) )

        self.start_to_record = 0
        self.human_intervent = 0
    
    def parse_events(self,agent,destination):
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 23:
                    self._autopilot_enabled = not self._autopilot_enabled
                    # 如果重新开启了自动驾驶，则重新设定目标路线，重新规划
                    self.start_to_record = 1  # 开始记录数据
                    if self._autopilot_enabled:
                        agent.set_destination(destination)
                if event.button == 0:     # 如果按下 "X" 键，则结束      
                    self.start_to_record = 0 # 停止记录数据
                    return True
                elif event.button == 4:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 2:
                    self.human_intervent = 1

            elif event.type == pygame.KEYDOWN:
                if event.type == pygame.QUIT:
                    self.start_to_record = 0 # 停止记录数据
                    return True
                elif event.key == pygame.K_ESCAPE:
                    self.start_to_record = 0 # 停止记录数据
                    return True
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    # 如果重新开启了自动驾驶，则重新设定目标路线，重新规划
                    if self._autopilot_enabled:
                        agent.set_destination(destination)                    

        if not self._autopilot_enabled:
            self._parse_vehicle_wheel()
            self._control.reverse = self._control.gear < 0
        return False
    
    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

class LowPassFilter:
    def __init__(self, cutoff_freq, sampling_freq, num_taps=5):
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq
        self.b, self.a = butter(num_taps, 2 * cutoff_freq / sampling_freq, 'low', analog=False)
        self.buffer = []

    def filter(self, new_data):
        self.buffer.append(new_data)
        filtered_data = lfilter(self.b, self.a, self.buffer)[-1]
        return filtered_data
    
def autopilot(vehicle,destination):
    # 输入对应车辆，目的地，返回路径点，控制等
    agent = BehaviorAgent(vehicle)
    route = agent.set_destination(destination,carla.Location(86.7620849609375, -6.435072422027588,11.9))
    
    return route ,agent

def draw_waypoint(route,world):
    for point in route:
        world.debug.draw_point(point[0].transform.location,
                                color = carla.Color(*color_bar[point[1]-1]),
                                life_time = 0)
    world.debug.draw_box(carla.BoundingBox( point[0].transform.location, carla.Vector3D(5,5,5)),
                        point[0].transform.rotation,
                        thickness=0.5)  
    
    x = point[0].transform.location.x
    y = point[0].transform.location.y
    z = point[0].transform.location.z
    
    world.debug.draw_arrow(carla.Location(x,y,z+10),
                           carla.Location(x,y,z+5),
                           arrow_size=0.5,)  

def carla_vec_to_np_array(vec):
    return np.array([vec.x,
                     vec.y,
                     vec.z])

def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False,throttle_limit = 1.0):
    throttle = np.clip(throttle, 0.0, throttle_limit)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)

def waypoint2traj(wps, vehicle,transform2vehicle = 1,contain_yaw = 0):
    '''
    Trasform waypoint to trajectory
    Parameters
    ----------
    wps : carla.Waypoint
        waypoint
    vehicle : carla.vehicle
        vechile from carla, trajectory will be planned base on its location
    transform2vehicle : bool,optional
        if true, will return world coordinates, else object coordinates
    contain_yaw : bool,optional
        if true, trajectory will contain yaw as last column
    
    Returns
    -------
    traj : ndarray
        trajectory, if contain_yaw is true ,it will be n*3 [x, y, yaw].
    '''
    if( transform2vehicle ):
        # transform waypoints to vehicle ref frame
        traj = np.array(
            [np.array([*carla_vec_to_np_array(x[0].transform.location), 1.]) for x in wps]
        ).T
        # 
        trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())

        traj = trafo_matrix_world_to_vehicle @ traj
        traj = traj.T
        traj = traj[:,:2]
    else:
        traj = np.array( [np.array([*carla_vec_to_np_array(x[0].transform.location)]) for x in wps] )
        traj = traj[:,:2]
    
    if( contain_yaw ):
        yaw = np.array( [x[0].transform.rotation.yaw for x in wps] )
        traj = np.concatenate( (traj,yaw.reshape(-1,1)),axis=1 )
    
    return traj    

def reach_destination(vehicle,destination,radius=5):
    """

    Parameters
    ----------
    vehicle : carla.Vehicle
        车辆对应api
    destination : carla.Location
        目的地
    radius : int, optional
        半径, by default 5

    Returns
    -------
    bool
        如果车辆位置处于目的地前后左右半径内，则返回true，否则返回false
    """
    x = vehicle.get_transform().location.x
    y = vehicle.get_transform().location.y
    if x <= destination.x+radius and x >= destination.x-radius and y <= destination.y+radius and y >= destination.y-radius:
        return True
    return False

def main(args=None):
    client = carla.Client('localhost', 2000)
    client.set_timeout(80.0)
    client.load_world('Town04')
    world = client.get_world()
        
    try:
        rclpy.init(args=args)
        main_node = MainNode(world)
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(main_node)

        spin_thread = Thread(target=main_node.spin)
        spin_thread.start()
        main_node.main_loop()
    finally:
        field_names = list(main_node.recorder.keys())
        rows = zip(*main_node.recorder.values())
        with open('recoder_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(field_names)
            for row in rows:
                writer.writerow(row)

        print('destroying actors.')
        for actor in main_node.actor_list:
            actor.destroy()
        spin_thread.join()
        pygame.quit()
        main_node.destroy_node()
        rclpy.shutdown()    
        print('done.')        

if __name__ == '__main__':
    main()