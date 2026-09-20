"""Render physical task goals without stepping or modifying the live environment."""

import numpy as np

GOAL_IMAGE_KEY = "goal_image"
GOAL_SOURCE = "physical_render_v1"


class PhysicalGoalRenderer:
    def __init__(self, env, domain, task, goal, size, camera):
        if goal.get("source") != GOAL_SOURCE:
            raise ValueError(f"Physical goals require source={GOAL_SOURCE}.")
        self.task_name = f"{domain}/{task}"
        fields = {
            "cartpole/balance_sparse": {"cart_position", "pole_angle"},
            "reacher/easy": {"elbow_sign"},
            "ball_in_cup/catch": {"cup_displacement", "ball_to_target"},
            "point_mass/easy": {"position"},
        }
        self.spec = dict(goal["observation"])
        if self.task_name not in fields or set(self.spec) != fields[self.task_name]:
            raise ValueError(f"Unexpected physical goal specification for {self.task_name}: {self.spec}")
        for value in self.spec.values():
            if not np.isfinite(np.asarray(value, dtype=float)).all():
                raise ValueError("Physical goals must be finite.")
        # The copy owns both model and data. Only Reacher's task target is refreshed.
        self.physics = env.physics.copy(share_model=False)
        self.task = env.task
        self.size, self.camera = size, camera
        self._target = None
        self._image = None

    def render(self, live_physics):
        target = None
        if self.task_name == "reacher/easy":
            target = np.array(live_physics.named.model.geom_pos["target"], copy=True)
        if self._image is not None and (target is None or np.array_equal(target, self._target)):
            return self._image

        physics, goal = self.physics, self.spec
        with physics.reset_context():
            if self.task_name == "cartpole/balance_sparse":
                physics.named.data.qpos["slider"] = float(goal["cart_position"])
                physics.named.data.qpos["hinge_1"] = float(goal["pole_angle"])
            elif self.task_name == "reacher/easy":
                physics.named.model.geom_pos["target"] = target
                sign = float(goal["elbow_sign"])
                if sign not in (-1, 1):
                    raise ValueError("Reacher elbow_sign must be -1 or 1.")
                l1 = float(physics.named.model.body_pos["hand", "x"])
                l2 = float(physics.named.model.body_pos["finger", "x"])
                x, y = target[:2] - physics.named.model.body_pos["arm", :2]
                cosine = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
                if not -1 <= cosine <= 1:
                    raise ValueError("Reacher target is outside the arm's workspace.")
                elbow = sign * np.arccos(cosine)
                shoulder = np.arctan2(y, x) - np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow))
                physics.named.data.qpos["shoulder"] = shoulder
                physics.named.data.qpos["wrist"] = elbow
            elif self.task_name == "ball_in_cup/catch":
                physics.named.data.qpos[["cup_x", "cup_z"]] = self._pair("cup_displacement")
            else:
                physics.data.qpos[:] = self._pair("position")

        if self.task_name == "ball_in_cup/catch":
            # Joint positions are displacements from the model's body origins.
            desired_ball = physics.named.data.site_xpos["target", ["x", "z"]] - self._pair("ball_to_target")
            displacement = desired_ball - physics.named.data.geom_xpos["ball", ["x", "z"]]
            physics.named.data.qpos[["ball_x", "ball_z"]] = displacement
            physics.after_reset()

        limited = physics.model.jnt_limited.astype(bool)
        positions = physics.data.qpos[physics.model.jnt_qposadr[limited]]
        bounds = physics.model.jnt_range[limited]
        if np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1]):
            raise ValueError("Physical goal violates joint limits.")
        if float(self.task.get_reward(physics)) < 1 - 1e-6:
            raise ValueError("Rendered physical goal does not satisfy the DMC task.")
        self._image = np.asarray(physics.render(*self.size, camera_id=self.camera), dtype=np.uint8)
        self._target = target
        return self._image

    def _pair(self, name):
        value = np.asarray(self.spec[name], dtype=float)
        if value.shape != (2,):
            raise ValueError(f"Physical goal {name} requires two coordinates.")
        return value

    def close(self):
        self.physics.free()
