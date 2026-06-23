import asyncio
import logging

import numpy as np
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import math

from astra_teleop.process import get_solve
from astra_teleop_web.webserver import WebServer

logger = logging.getLogger(__name__)

GRIPPER_MAX = 0.055
INITIAL_LIFT_DISTANCE = 0.916

FAR_SEEING_HEAD_TILT = 0.26

class Teleopoperator:
    def __init__(self):        
        self.webserver = WebServer()
        self.webserver.on_hand = self.hand_cb
        self.webserver.on_pedal = self.pedal_cb
        self.webserver.on_control = self.control_cb
        self.webserver.on_remote = self.remote_cb
        
        self.on_pub_goal = None
        self.on_pub_gripper = None
        self.on_pub_head = None
        self.on_cmd_vel = None
        
        self.on_get_current_eef_pose = None
        self.on_get_initial_eef_pose = None
        self.on_reset = None
        self.on_done = None
        self.on_rerecord = None
        
        self.teleop_mode = None
        self.remote_mode = "gripper"
        
        self.percise_mode = True
        self.solve = get_solve(scale=1.0)
        
        self.lift_distance = INITIAL_LIFT_DISTANCE
        self.Tscam = { "left": None, "right": None, }
        self.Tcamgoal_last = { "left": None, "right": None }
        
        self.gripper_lock = { "left": False, "right": False }
        self.last_gripper_pos = { "left": GRIPPER_MAX, "right": GRIPPER_MAX }
        
        self.far_seeing = False
        
    async def reset_Tscam(self):
        self.Tscam = { "left": None, "right": None, }
        self.Tcamgoal_last = { "left": None, "right": None }
        
        while True: # wait for new tag result
            ok = True
            for side in ["left", "right"]:
                if self.Tcamgoal_last[side] is None:
                    # logger.info(f"Waiting for new Tcamgoal_last {side}")
                    ok = False
            if ok:
                break
            
            await asyncio.sleep(0.1)
            
        for side in ["left", "right"]:
            Tsgoal = self.on_get_current_eef_pose(side)
            Tcamgoal = self.Tcamgoal_last[side]
            self.Tscam[side] = Tsgoal @ np.linalg.inv(Tcamgoal)
            logger.info(f"Tscam ({side}): \n{str(self.Tscam[side])}")
                
    async def update_percise_mode(self, percise_mode):
        last_percise_mode = self.percise_mode
        # if last_percise_mode == percise_mode:
        #     return
        self.percise_mode = percise_mode
        self.solve = get_solve(scale=0.5 if self.percise_mode == "more_percise" else 1.0) # scale means to amplify motion
        await self.reset_Tscam()
        
    def update_teleop_mode(self, teleop_mode):
        self.teleop_mode = teleop_mode
        assert self.teleop_mode in ["on", None]
        if self.teleop_mode == "on":
            if self.percise_mode == "more_percise":
                self.webserver.control_datachannel_log("Teleop Mode: On (More Percise)")
                logger.info("Teleop Mode: On (More Percise)")
            elif self.percise_mode:
                self.webserver.control_datachannel_log("Teleop Mode: On (Percise)")
                logger.info("Teleop Mode: On (Percise)")
            else:
                self.webserver.control_datachannel_log("Teleop Mode: On")
                logger.info("Teleop Mode: On")
        else:
            self.webserver.control_datachannel_log("Teleop Mode: Off")
            logger.info("Teleop Mode: Off")

    async def reset_arm(self, lift_distance=INITIAL_LIFT_DISTANCE, joint_bent=math.pi/4, far_seeing=False):  
        self.far_seeing = far_seeing
        self.lift_distance = lift_distance
        goal_pose = {
            "left": self.on_get_initial_eef_pose("left", [self.lift_distance, joint_bent, -joint_bent, 0, 0, 0]),
            "right": self.on_get_initial_eef_pose("right", [self.lift_distance, -joint_bent, joint_bent, 0, 0, 0]),
        }

        while True:
            ok = True
            for side in ["left", "right"]:
                curr_pose = self.on_get_current_eef_pose(side)
                
                goal_pose_pq = pt.pq_from_transform(goal_pose[side])
                curr_pose_pq = pt.pq_from_transform(curr_pose)
                
                pos_dist = math.dist(goal_pose_pq[:3], curr_pose_pq[:3])
                rot_dist = pr.quaternion_dist(
                    goal_pose_pq[3:],
                    curr_pose_pq[3:]
                )
            
                if not (pos_dist < 0.02 and rot_dist < 0.03):
                    # logger.info(f"Resetting {side}: pos_dist {pos_dist}m, rot_dist {rot_dist}rad, curr_pose: \n{curr_pose}")
                    ok = False            
            if ok:
                break
            
            for side in ["left", "right"]:
                self.on_pub_goal(side, goal_pose[side])
                self.on_pub_gripper(side, self.last_gripper_pos[side])
            
            if self.far_seeing:
                self.on_pub_head(0, FAR_SEEING_HEAD_TILT)
            else:
                self.on_pub_head(0, self.get_head_tilt(self.lift_distance))
            
            await asyncio.sleep(0.1)

    def get_head_tilt(self, lift_distance):
        point0_lift = 0
        point0_tilt = 1.36
        point1_lift = 0.8
        point1_tilt = 1.06
        
        return point0_tilt + (point1_tilt - point0_tilt) * (lift_distance - point0_lift) / (point1_lift - point0_lift)

    def hand_cb(self, camera_matrix, distortion_coefficients, corners, ids):
        Tcamgoal = {}

        Tcamgoal["left"], Tcamgoal["right"] = self.solve(
            camera_matrix, distortion_coefficients,
            aruco_corners=corners, aruco_ids=ids,
            debug=False,
            debug_image=None,
        ) # 1ms@1080p
        
        for side in ["left", "right"]:
            if Tcamgoal[side] is not None:
                if self.Tcamgoal_last[side] is None:
                    self.Tcamgoal_last[side] = Tcamgoal[side]

                if self.percise_mode:
                    # low_pass_coff = 0.1
                    # Tcamgoal[side] = pt.transform_from_pq(pt.pq_slerp(
                    #     pt.pq_from_transform(self.Tcamgoal_last[side]),
                    #     pt.pq_from_transform(Tcamgoal[side]),
                    #     low_pass_coff
                    # ))

                    # trust for sensor read (in this case, opencv on web client)
                    p_low_pass_coff = 0.1
                    q_low_pass_coff = 0.1
                    pq_camgoal_last = pt.pq_from_transform(self.Tcamgoal_last[side])
                    pq_camgoal = pt.pq_from_transform(Tcamgoal[side])
                    p = pq_camgoal_last[:3] * (1 - p_low_pass_coff) + pq_camgoal[:3] * p_low_pass_coff
                    q = pr.quaternion_slerp(pq_camgoal_last[3:], pq_camgoal[3:], q_low_pass_coff, shortest_path=True)
                    Tcamgoal[side] = pt.transform_from_pq(np.concatenate([p, q]))

                self.Tcamgoal_last[side] = Tcamgoal[side]

        if self.teleop_mode is not None:
            for side in ["left", "right"]:
                if self.Tcamgoal_last[side] is None:
                    self.webserver.control_datachannel_log(f"Connect capture and making sure {side} are in the camera view!")
                    raise Exception(f"Connect capture and making sure {side} are in the camera view!")

            for side in ["left", "right"]:
                if self.Tscam[side] is None:
                    self.webserver.control_datachannel_log(f"Reset {side} arm first!")
                    raise Exception(f"Reset {side} arm first!")
            
            for side in ["left", "right"]:
            # for side in ["left"]:
                Tsgoal = self.Tscam[side] @ self.Tcamgoal_last[side]
                self.on_pub_goal(side, Tsgoal, Tscam=self.Tscam[side], Tsgoal_inactive=Tsgoal)
        
            if self.far_seeing:
                self.on_pub_head(0, FAR_SEEING_HEAD_TILT)
            else:
                self.on_pub_head(0, self.get_head_tilt(self.lift_distance))
        
    def pedal_cb(self, pedal_real_values):
        pedal_names = ["angular-pos", "angular-neg", "linear-neg", "linear-pos"]
        non_sensetive_area = 0.1
        cliped_pedal_real_values = np.clip((np.array(pedal_real_values) - 0.5) / (0.5 - non_sensetive_area) * 0.5 + 0.5, 0, 1)
        
        # Debug Log: Print values periodically to check input range
        # if np.random.rand() < 0.05: # approx every 20 calls
            # logger.info(f"Pedal Raw: {np.round(pedal_real_values, 2)} -> Clipped: {np.round(cliped_pedal_real_values, 2)}")
            
        # if self.teleop_mode == "arm":
        # elif self.teleop_mode == "base":
        values = dict(zip(pedal_names, cliped_pedal_real_values))

        LINEAR_VEL_MAX = 1
        ANGULAR_VEL_MAX = 1
        linear_vel = (values["linear-pos"] - values["linear-neg"]) * LINEAR_VEL_MAX
        angular_vel = (values["angular-pos"] - values["angular-neg"]) * ANGULAR_VEL_MAX * (-1 if linear_vel < 0 else 1)

        self.on_cmd_vel(linear_vel, angular_vel)

    def remote_cb(self, remote_values):
        side = remote_values["side"]
        force = remote_values.get("force", 0.0)
        # event_code = remote_values.get("eventCode", 0)
        # button_id = remote_values.get("buttonId", 0)
        
        if self.teleop_mode is None:
            return

        if side not in ["left", "right"]:
            return
        
        if self.remote_mode == "lift":
            # 右升左降
            LIFT_VEL_MAX = 0.5
            lift_vel = force * LIFT_VEL_MAX * (1 if side == "right" else -1)

            TIME_DELTA = 0.1 # TODO Better solution
            change = lift_vel * TIME_DELTA
            change_left = 1
            change_right = 1
            LIFT_DISTANCE_MIN = 0
            LIFT_DISTANCE_MAX = 1.2
            if self.lift_distance + change < LIFT_DISTANCE_MIN or self.lift_distance + change > LIFT_DISTANCE_MAX:
                change_left = 0 if self.lift_distance + change < LIFT_DISTANCE_MIN else 1
                change_right = 0 if self.lift_distance + change > LIFT_DISTANCE_MAX else 1
                logger.warn("Lift Over Limit")
                self.webserver.control_datachannel_log("Lift Over Limit")
            elif change:
                self.Tscam["left"][2,3] += change*change_left
                self.Tscam["right"][2,3] += change*change_right
                self.lift_distance += change
                logger.info(f"Lift Distance: {self.lift_distance:.3f}")
                self.webserver.control_datachannel_log(f"Lift Distance: {self.lift_distance:.3f}")
        elif self.remote_mode == "gripper":
            gripper_pos = (1 - force) * GRIPPER_MAX
            
            if self.gripper_lock[side] == True and gripper_pos > GRIPPER_MAX * 0.9:
                self.gripper_lock[side] = 'ready_to_unlock'
                
                logger.info(f"{side.capitalize()} Gripper Lock: Locked (Ready to Unlock)")
                self.webserver.control_datachannel_log(f"{side.capitalize()} Gripper Lock: Locked (Ready to Unlock)")
                
            if self.gripper_lock[side] == 'ready_to_unlock' and gripper_pos <= self.last_gripper_pos[side]:
                self.gripper_lock[side] = False
                logger.info(f"{side.capitalize()} Gripper Lock: Unlocked")
                self.webserver.control_datachannel_log(f"{side.capitalize()} Gripper Lock: Unlocked")
            
            if self.gripper_lock[side] == False:
                self.last_gripper_pos[side] = gripper_pos
                self.on_pub_gripper(side, self.last_gripper_pos[side])
                
        else:
            logger.warn(f"Unknown remote mode: {self.remote_mode}")
            self.webserver.control_datachannel_log(f"Unknown remote mode: {self.remote_mode}")
    
    async def control_cb(self, control_type):
        self.webserver.control_datachannel_log(f"Cmd: {control_type}")
        logger.info(f"Cmd: {control_type}")
        if control_type == "operate_reset":
            self.update_teleop_mode(None)
            self.last_gripper_pos = { "left": GRIPPER_MAX, "right": GRIPPER_MAX }
            await self.reset_arm(INITIAL_LIFT_DISTANCE, math.pi/4, far_seeing=False)

            # # uncomment when collecting datas to avoid stair in the dataset
            # await self.update_percise_mode(percise_mode=True) 
            # self.update_teleop_mode("arm")
            
            self.on_reset()
            self.webserver.control_datachannel_log("Reset event")
            logger.info("Reset event")
        elif control_type == "done":
            self.on_done()
            self.webserver.control_datachannel_log("Done event")
            logger.info("Done event")
        elif control_type == "rerecord":
            self.on_rerecord()
            self.webserver.control_datachannel_log("Rerecord event")
            logger.info("Rerecord event")
        elif control_type == "teleop_mode_off":
            self.update_teleop_mode(None)
        elif control_type == "teleop_mode_on":
            await self.update_percise_mode(percise_mode=True)
            self.update_teleop_mode("on")
        elif control_type == "teleop_mode_toggle":
            if self.teleop_mode is None:
                await self.update_percise_mode(percise_mode=True)
                self.update_teleop_mode("on")
            else:
                self.update_teleop_mode(None)
        elif control_type == "move_reset":
            self.update_teleop_mode(None)
            await self.reset_arm(self.lift_distance, math.pi/2*0.9, far_seeing=True)
            await self.update_percise_mode(percise_mode=True)
        elif control_type == "percise_mode_false":
            self.update_teleop_mode(None)
            await self.update_percise_mode(percise_mode=False)
            self.update_teleop_mode("on")
        elif control_type == "percise_mode_true":
            self.update_teleop_mode(None)
            await self.update_percise_mode(percise_mode=True)
            self.update_teleop_mode("on")
        elif control_type == "percise_mode_more_percise":
            self.update_teleop_mode(None)
            await self.update_percise_mode(percise_mode="more_percise")
            self.update_teleop_mode("on")
        elif control_type == "percise_mode_next":
            next_mode_dict = {
                False: True,
                True: "more_percise",
                "more_percise": False,
            }
            self.update_teleop_mode(None)
            await self.update_percise_mode(percise_mode=next_mode_dict[self.percise_mode])
            self.update_teleop_mode("on")
        elif control_type == "gripper_lock_left":
            self.gripper_lock["left"] = True
            logger.info("Left Gripper Lock: Locked, release your pedal to unlock")
            self.webserver.control_datachannel_log("Left Gripper Lock: Locked, release your pedal to unlock")
        elif control_type == "gripper_lock_right":
            self.gripper_lock["right"] = True
            logger.info("Right Gripper Lock: Locked, release your pedal to unlock")
            self.webserver.control_datachannel_log("Right Gripper Lock: Locked, release your pedal to unlock")
        elif control_type == "gripper_mode":
            self.remote_mode = "gripper"
            logger.info("Change to Gripper Mode")
            self.webserver.control_datachannel_log("Change to Gripper Mode")
        elif control_type == "lift_mode":
            self.remote_mode = "lift"
            logger.info("Change to Lift Mode")
            self.webserver.control_datachannel_log("Change to Lift Mode")
        elif control_type == "remote_mode_toggle":
            self.remote_mode = "lift" if self.remote_mode == "gripper" else "gripper"
            if self.remote_mode == "gripper":
                logger.info("Change to Gripper Mode")
                self.webserver.control_datachannel_log("Change to Gripper Mode")
            else:
                logger.info("Change to Lift Mode")
                self.webserver.control_datachannel_log("Change to Lift Mode")
            
    def error_cb(self, msg):
        self.webserver.loop.call_soon_threadsafe(self.webserver.control_datachannel_log, msg)
