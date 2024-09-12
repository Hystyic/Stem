import bpy

def create_headset():
  # Create a cylinder primitive.
  cylinder = bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1)

  # Scale the cylinder so that it is the size of the headset you want to create.
  cylinder.scale = (1, 1, 0.5)

  # Add a Subdivision Surface modifier to the cylinder.
  cylinder.modifiers.new("Subdivision Surface", type="SUBSURF")
  cylinder.modifiers["Subdivision Surface"].iterations = 3

  # Extrude the top face of the cylinder to create the ear cups.
  bpy.ops.mesh.extrude_region_to_cursor(enter_editmode=False)

  # Add a Bevel modifier to the ear cups.
  bpy.ops.object.modifier_add(type="BEVEL")
  bpy.context.object.modifiers["Bevel"].width = 0.1
  bpy.context.object.modifiers["Bevel"].depth = 0.05

  # Extrude the bottom face of the cylinder to create the headband.
  bpy.ops.mesh.extrude_region_to_cursor(enter_editmode=False)

  # Add a Bevel modifier to the headband.
  bpy.ops.object.modifier_add(type="BEVEL")
  bpy.context.object.modifiers["Bevel"].width = 0.1
  bpy.context.object.modifiers["Bevel"].depth = 0.05

  # Add a bone conduction speaker to the ear cups.
  speaker = bpy.ops.mesh.primitive_cone_add(radius=0.2, depth=0.2)
  speaker.location = (0, 0, 0.1)

  # Rename the objects so that you can easily identify them.
  bpy.context.object.name = "Ear Cup"
  bpy.context.object.children[0].name = "Headband"
  bpy.context.object.children[1].name = "Speaker"

  # Position and rotate the objects so that they are arranged like a headset.
  bpy.context.object.location = (0, 0, 0)
  bpy.context.object.rotation_euler = (0, 0, 90)
  bpy.context.object.children[0].location = (0, 0, -1)
  bpy.context.object.children[1].location = (0, 0, 0.1)

  # Apply the modifiers to the objects.
  bpy.ops.object.modifier_apply(modifier="Subdivision Surface")
  bpy.ops.object.modifier_apply(modifier="Bevel")

  return bpy.context.object


if __name__ == "__main__":
  create_headset()
