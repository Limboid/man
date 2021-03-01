from magnebot import Magnebot, Arm

m = Magnebot()
print('m')

# Initialize the scene, populate it with objects, and add the Magnebot.
# This can take a few minutes to finish.
m.init_scene(scene="1a", layout=0, room=1)
print('init scene')

# Reach for a target position.
status = m.reach_for(arm=Arm.left, target={"x": 0.1, "y": 0.6, "z": 0.4}, absolute=False)
print(status) # ActionStatus.success

# Save images.
m.state.save_images(output_directory="magnebot_test_images")

# End the simulation.
m.end()