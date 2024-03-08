import os

image_size = (32, 32)
save_directory = "checkpoints"
beam_width = 16
num_characters = 128

# Load the classes if they exist.
file = os.path.join(save_directory, "classes.txt")
if os.path.exists(file):
    with open(file) as f:
        classes = [int(x) for x in f.read().splitlines()]
else:
    classes = list(range(num_characters))
