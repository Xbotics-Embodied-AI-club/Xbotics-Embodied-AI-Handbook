'''

Overriding the so100 to add an is_touching() function for compatibility.
Note: This only works with "third" camera, "wrist" camera needs some modifications (gripper link setting and naming).

'''

import torch

from mani_skill.agents.robots.so100.so_100 import SO100
from mani_skill.utils.structs.actor import Actor


def _is_touching(self, object: Actor):
    """Check if the robot is touching an object"""
    l_contact_forces = self.scene.get_pairwise_contact_forces(
        self.finger1_link, object
    )
    r_contact_forces = self.scene.get_pairwise_contact_forces(
        self.finger2_link, object
    )
    lforce = torch.linalg.norm(l_contact_forces, axis=1)
    rforce = torch.linalg.norm(r_contact_forces, axis=1)
    return torch.logical_or(lforce >= 1e-2, rforce >= 1e-2)


# Monkey-patch the method onto the original class
SO100.is_touching = _is_touching
