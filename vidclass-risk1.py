#-------------------------------------------------------------------------------
# Name:        LavalampParticles.py
# Purpose: pretty visualizer
#
# Author:      The Schim
#
# Created:     16/03/2023
# Copyright:   (c) The Schim 2023
# Licence:     CC0
#-------------------------------------------------------------------------------
import pygame
import random
import math
from colorsys import hsv_to_rgb, rgb_to_hsv
import uuid
import cv2
import pandas as pd
import csv
import numpy as np
import pyautogui

# Constants
WIDTH, HEIGHT = 800, 600
BG_COLOR = (0, 0, 0)
FPS = 60
MIN_RADIUS = 33.3
MAX_RADIUS = 99.9
SPLIT_PROB = 0.29
DEPTH = 700
cooldown = random.randint(314, 6400)
INITIAL_GLOBS = 50
MAX_NUMBER_GLOBS = 230
SPEED_DIVISOR = 2.0+(1/math.pi)
AGE_FACTOR = 0.1
TRANSFER = 0.00075
COLOR_CATEGORIES = {
    "red": (0, 255, 0),
    "red-orange": (22, 234, 117),
    "orange": (56, 224, 0),
    "yellow-orange": (84, 213, 0),
    "yellow": (127, 191, 0),
    "lime": (191, 127, 0),
    "green": (224, 56, 0),
    "teal": (234, 22, 117),
    "blue": (255, 0, 0),
    "indigo": (217, 0, 92),
    "purple": (170, 0, 85),
    "magenta": (85, 0, 170),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "white": (255, 255, 255)
}


def random_point_on_ellipsoid(a, b, c):
    while True:
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        w = random.uniform(-1, 1)
        d = u**2/a**2 + v**2/b**2 + w**2/c**2

        if d <= 1:
            break

    x = (WIDTH / 2) + a * u
    y = (HEIGHT / 2) + b * v
    z = (DEPTH / 2) + c * w

    x = max(MIN_RADIUS, min(WIDTH - MIN_RADIUS, x))
    y = max(MIN_RADIUS, min(HEIGHT - MIN_RADIUS, y))
    z = max(MIN_RADIUS, min(DEPTH - MIN_RADIUS, z))

    return x, y, z

def color_difference(color1, color2):
    return sum(abs(color1[i] - color2[i]) for i in range(3))

def is_similar_color(color1, color2, threshold=32):
    return color_difference(color1, color2) < threshold

def calculate_mutation_range(globs):
    total_globs = len(globs)
    similar_color_count = 0

    for i in range(total_globs):
        for j in range(i+1, total_globs):
            if is_similar_color(globs[i].color, globs[j].color):
                similar_color_count += 1

    percentage_similar_color = similar_color_count / total_globs
    mutation_range = int(percentage_similar_color * 255)

    return mutation_range

def wild_color_mutation(parent_color, mutation_range):
    mutated_color = tuple(
        max(64, min(255, parent_color[i] + random.randint(-mutation_range, mutation_range)))
        for i in range(3)
    )
    return mutated_color

# Add a helper function to lerp between two values
def lerp(a, b, t):
    return a + (b - a) * t

class Glob:
    def __init__(self, x, y, z, radius, color, set_id=None, glob_sets=None):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.color = color
        self.glob_sets = glob_sets if glob_sets is not None else {}  # set default value
        self.creation_time = pygame.time.get_ticks()
        self.milestone1 = self.color
        self.milestone2 = self._get_next_milestone(self.color)
        self.lerp_t = 0
        self.lerp_speed = 0.0084

        if set_id is None:
            set_id = str(uuid.uuid4())

        self.set_id = set_id

        if self.set_id not in self.glob_sets:
            self.glob_sets[self.set_id] = set()
        self.glob_sets[self.set_id].add(self)

        speed_multiplier = 28.88 / self.radius
        self.vx = (random.uniform(-1, 1) / speed_multiplier) / SPEED_DIVISOR
        self.vy = (random.uniform(-1, 1) / speed_multiplier) / SPEED_DIVISOR
        self.vz = (random.uniform(-1, 1) / speed_multiplier) / SPEED_DIVISOR

        if self.radius == MAX_RADIUS:
            self.num_globs = len(INITIAL_GLOBS)
        else:
            self.num_globs = round(self.radius / (MAX_RADIUS / INITIAL_GLOBS))

        self.split_prob = SPLIT_PROB

    def _get_next_milestone(self, current_color):
        next_color = []
        for channel in current_color:
            min_val = max(0, channel - 128)
            max_val = min(255, channel + (255 - channel))
            next_channel = random.randint(min_val, max_val)
            next_color.append(next_channel)
        return tuple(next_color)

    def split(self, globs):
        if len(globs) < MAX_NUMBER_GLOBS and random.random() < self.split_prob:
            new_globs = []
            num_new_globs = random.randint(round(2*((self.radius/MAX_RADIUS*0.5)+1)), round(5*((self.radius/MAX_RADIUS*0.5)+1)))
            for _ in range(num_new_globs):
                new_x = self.x + random.uniform(-self.radius, self.radius)
                new_y = self.y + random.uniform(-self.radius, self.radius)
                new_z = self.z + random.uniform(-self.radius, self.radius)
                new_radius = self.radius / num_new_globs

                # Use wild color mutation for offspring
                mutation_range = calculate_mutation_range(globs)
                new_color = wild_color_mutation(self.color, mutation_range)

                new_glob = Glob(new_x, new_y, new_z, new_radius, new_color, self.set_id, self.glob_sets)
                new_glob.split_prob = self.split_prob
                new_globs.append(new_glob)
            return new_globs
        else:

            return None

    def draw(self, screen, bg_color):
        scale_factor = get_scale_factor(self.z, DEPTH)

        x = self.x * scale_factor + (1 - scale_factor) * (WIDTH / 2)
        y = self.y * scale_factor + (1 - scale_factor) * (HEIGHT / 2)
        scaled_radius = int(self.radius * scale_factor)

        r = int(self.color[0] * scale_factor + bg_color[0] * (1 - scale_factor))
        g = int(self.color[1] * scale_factor + bg_color[1] * (1 - scale_factor))
        b = int(self.color[2] * scale_factor + bg_color[2] * (1 - scale_factor))
        fade_color = (r, g, b)

        # Ensure fade_color is a valid RGB tuple
        fade_color = tuple(max(0, min(c, 255)) for c in fade_color)

        pygame.draw.circle(screen, fade_color, (int(x), int(y)), scaled_radius)

    def update(self, globs, glob_sets):
        global TRANSFER
        removed = False
        # Move glob according to its speed
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz

        # Apply boundary conditions
        self.x %= WIDTH
        self.y %= HEIGHT
        self.z %= DEPTH

        # Update color according to the current milestones
        if self.lerp_t < 1:
            self.color = tuple(int(lerp(self.milestone1[i], self.milestone2[i], self.lerp_t)) for i in range(3))
            self.lerp_t += self.lerp_speed
        else:
            self.milestone1 = self.milestone2
            self.milestone2 = self._get_next_milestone(self.color)
            self.lerp_t = 0

        # Move globs out of sibling set if they are far enough apart
        siblings = [g for g in self.glob_sets[self.set_id] if g != self]
        for sibling in siblings:
            distance = math.sqrt((self.x - sibling.x)**2 + (self.y - sibling.y)**2 + (self.z - sibling.z)**2)
            if distance > 2 * self.radius:
                self.glob_sets[self.set_id].remove(self)
                new_set_id = str(uuid.uuid4())
                self.set_id = new_set_id
                self.glob_sets[new_set_id] = {self}
                break

        # Handle glob collision and color blending
        for other in globs:
            if other != self:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
                if distance <= self.radius + other.radius:
                    if self.radius > other.radius:
                        larger, smaller = self, other
                    else:
                        larger, smaller = other, self

                    transfer_rate = TRANSFER  # Adjust this value to control the transfer rate
                    transferred_radius = smaller.radius * transfer_rate
                    larger.radius += transferred_radius
                    smaller.radius -= transferred_radius

                    # Color blending
                    larger_area = math.pi * larger.radius**2
                    smaller_area = math.pi * smaller.radius**2
                    total_area = larger_area + smaller_area
                    new_color = tuple(int((larger_area * larger.color[i] + smaller_area * smaller.color[i]) / total_area) for i in range(3))
                    larger.color = new_color

                    # Remove smaller glob if its radius becomes zero
                    if smaller.radius <= 0:
                        globs.remove(smaller)
                        if smaller.set_id in glob_sets and smaller in glob_sets[smaller.set_id]:
                            glob_sets[smaller.set_id].remove(smaller)
                            self.num_globs -= 1 # decrement the num_globs of the parent glob
                        removed = True
                        break

        # Check if the glob should split, outside the loop
        if self.radius > MAX_RADIUS:
            new_globs = self.split(globs)
            if new_globs:
                globs.extend(new_globs)
                if not removed and self in globs:
                    globs.remove(self)
                    self.num_globs -= 1 # decrement the num_globs of the parent glob

def attract_smaller_globs(globs, min_radius):
    force = 0.3/4.6
    for glob1 in globs:
        if glob1.radius < min_radius:
            nearest_larger_glob = None
            nearest_distance = float('inf')
            for glob2 in globs:
                if glob2.radius >= min_radius and glob2 != glob1:
                    distance = math.sqrt((glob1.x - glob2.x) ** 2 + (glob1.y - glob2.y) ** 2 + (glob1.z - glob2.z) ** 2)
                    if distance < nearest_distance:
                        nearest_larger_glob = glob2
                        nearest_distance = distance
            if nearest_larger_glob is not None:
                attraction_force = force * (min_radius / nearest_distance)
                dx = nearest_larger_glob.x - glob1.x
                dy = nearest_larger_glob.y - glob1.y
                dz = nearest_larger_glob.z - glob1.z
                norm = math.sqrt(dx**2 + dy**2 + dz**2)
                glob1.vx += dx / norm * attraction_force
                glob1.vy += dy / norm * attraction_force
                glob1.vz += dz / norm * attraction_force

def get_attraction_force(color1, color2):
    h1, s1, v1 = rgb_to_hsv(*(c / 255 for c in color1))
    h2, s2, v2 = rgb_to_hsv(*(c / 255 for c in color2))

    hue_diff = abs(h1 - h2)
    saturation_diff = abs(s1 - s2)

    attraction_strength = (1 - hue_diff) * (1 - saturation_diff)
    attraction_force = 0.0002 * attraction_strength

    return attraction_force

def get_scale_factor(z, depth):
    return 1 - (z / depth)

def average_glob_hsv(globs):
    if len(globs) == 0:
        return (0, 0, 0)  # default background color if there are no globs

    num_globs = len(globs)
    total_h, total_s, total_v = 0, 0, 0
    for glob in globs:
        h, s, v = rgb_to_hsv(*(c / 255 for c in glob.color))
        total_h += h
        total_s += s
        total_v += v

    avg_h = total_h / num_globs
    avg_s = total_s / num_globs
    avg_v = total_v / num_globs

    return avg_h, avg_s, avg_v

def get_random_color():
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    return (r, g, b)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Nia.S & ChatGPT's Lavalamp")
    clock = pygame.time.Clock()

    a, b, c = WIDTH / 2, HEIGHT / 2, DEPTH / 2

    globs = [Glob(*random_point_on_ellipsoid(a, b, c),
                  random.uniform(MIN_RADIUS, MAX_RADIUS),
                  get_random_color(),
                  str(uuid.uuid4())) for _ in range(INITIAL_GLOBS)]

    # Initialize the glob sets with the initial globs
    glob_sets = {i: {glob} for i, glob in enumerate(globs)}

    running = True
    last_valid_bg_color = None  # initialize last_valid_bg_color variable
    frame_count = 0  # initialize frame_count variable
    color_counts = {color: 0 for color in COLOR_CATEGORIES}  # initialize color_counts dictionary

    # Open a CSV file for writing
    with open("Circle-count.csv", "w") as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row
        header_row = ["Frame", "Image", "Text"]
        for color in COLOR_CATEGORIES:
            header_row.append("Number of " + color + " circles")
        writer.writerow(header_row)

        while running:
            # Update background color
            try:
                avg_h, avg_s, avg_v = average_glob_hsv(globs)
                bg_color = tuple(int(c * 255) for c in hsv_to_rgb(1 - avg_h, 1 - avg_s, 1 - avg_v))
                screen.fill(bg_color)
                last_valid_bg_color = bg_color
            except ValueError:
                if last_valid_bg_color is not None:
                    screen.fill(last_valid_bg_color)
                else:
                    r, g, b = bg_color
                    avg_value = (r + g + b) // 3
                    default_color = (64, 64, 64) if avg_value >= 128 else (255, 255, 255)
                    screen.fill(default_color)
            globs.sort(key=lambda glob: glob.position.z, reverse=True)
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        if screen.get_flags() & pygame.FULLSCREEN:
                            pygame.display.set_mode((WIDTH, HEIGHT))
                        else:
                            pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

            new_globs = []

            # Update color counts
            color_counts = {color: 0 for color in COLOR_CATEGORIES}
            for glob in globs:
                color_counts[glob.color_category] += 1

            # Write row to CSV
            row = [frame_count, "image_" + str(frame_count) + ".png", "text_" + str(frame_count)]
            for color in COLOR_CATEGORIES:
                row.append(color_counts[color])
            writer.writerow(row)

            # Save screenshot of the frame
            pygame.image.save(screen, "image_" + str(frame_count) + ".png")

            # Update frame count
            frame_count += 1

            # Add new globs to the list
            for glob_set in glob_sets.values():
                new_globs += create_new_globs(glob_set)
            globs += new_globs

            # Update glob sets
            glob_sets = update_glob_sets(globs)

            # Draw globs and update their positions
            for glob in globs:
                glob.update_position(WIDTH, HEIGHT, DEPTH)
                globs.sort(key=lambda glob: glob.position.z, reverse=True)
                glob.draw(screen)

            # Check for collisions and merge globs
            collisions = find_collisions(globs)
            for glob1, glob2 in collisions:
                merged_glob = merge_globs(glob1, glob2)
                new_globs.append(merged_glob)

            # Add new globs to glob list and glob sets
            globs += new_globs
            for glob in new_globs:
                add_glob_to_sets(glob, glob_sets)

            # Remove merged globs from glob list and glob sets
            for glob1, glob2 in collisions:
                globs.remove(glob1)
                globs.remove(glob2)
                remove_glob_from_sets(glob1, glob_sets)
                remove_glob_from_sets(glob2, glob_sets)

            # Save the current state of the simulation to a PNG file
            if SAVE_FRAMES:
                pygame.image.save(screen, f"frames/{frame_count:05}.png")

            # Count the number of globs in each color category
            color_counts = {color: 0 for color in COLOR_CATEGORIES}
            for glob in globs:
                for color in COLOR_CATEGORIES:
                    if glob.color == color:
                        color_counts[color] += 1

            # Write a row to the CSV file for the current frame
            row = [frame_count, "frames/{:05d}.png".format(frame_count), "This is a sample text"]
            for color in COLOR_CATEGORIES:
                row.append(color_counts[color])
            writer.writerow(row)

            # Update frame count
            frame_count += 1

            # Update display
            pygame.display.flip()
            clock.tick(FPS)

    # Quit pygame
    pygame.quit()
if __name__ == "__main__":
    main()