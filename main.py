import pygame
import math
import sys
import numpy as np
import pygame.sprite
from neural_network import NeuralNetwork, DataPoint

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
TRACK = pygame.image.load("./circle_track.png").convert()

class Car(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()
        self.initial_image = pygame.image.load("./car.png") # image sourced from here: https://www.clipartmax.com/download/m2i8H7K9i8m2Z5Z5_f1-racing-car-icon-top-view-top-down-f1-car/
        self.image = self.initial_image
        self.rect = self.image.get_rect(center=(220, 270))
        self.angle = 33
        self.velocity = pygame.math.Vector2(100, 0)
        self.velocity.rotate_ip(-self.angle)
        self.pos = pygame.math.Vector2(self.rect.center)
        self.rotation_velocity = 3
        self.direction = 0
        self.alive = True
        self.radars = []
        self.time_alive = 0
        self.prev_pos = self.pos
        self.max_radar_distance = 100
        self.distance_traveled = 0
        self.speed = 1


    def update(self, dt, clock):
        if self.alive:
            self.radars.clear()
            self.drive(dt, clock)
            self.rotate()
            for radar_angle in (-45, 0, 45):
                self.radar(radar_angle)
            self.collision()
            return self.data()


    def drive(self, dt, clock):
        self.pos += self.velocity * dt
        self.rect.center = (round(self.pos.x), round(self.pos.y))
        self.time_alive += pygame.time.get_ticks() / 10000

        self.distance_traveled += self.speed + 1000 * dt


    def rotate(self):
        if self.direction == 1:
            # right
            self.angle -= self.rotation_velocity
            self.velocity.rotate_ip(self.rotation_velocity)
        if self.direction == -1:
            # left
            self.angle += self.rotation_velocity
            self.velocity.rotate_ip(-self.rotation_velocity)

        self.image = pygame.transform.rotozoom(self.initial_image, self.angle, 0.03)
        self.rect = self.image.get_rect(center=self.rect.center)


    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x,y)) == pygame.Color(0, 56, 0, 255) and length < self.max_radar_distance:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255,255,255,255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
        self.radars.append([radar_angle, dist])


    def collision(self):
        length = 28
        left_collision = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 25)) * length), int(self.rect.center[1] - math.sin(math.radians(self.angle + 25)) * length)]
        right_collision = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 25)) * length), int(self.rect.center[1] - math.sin(math.radians(self.angle - 25)) * length)]

        if SCREEN.get_at(right_collision) == pygame.Color(0, 56, 0, 255) or SCREEN.get_at(left_collision) == pygame.Color(0, 56, 0, 255):
            self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), right_collision, 2)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), left_collision, 2)

        
    def data(self):
        return [int(radar[1]) for radar in self.radars]


def paused():
    paused = True
    while paused:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    paused = False
                    break


def normalize_elements_in_array(array):
    if len(array) == 1:
        return [1]
    minimum = min(array)
    maximum = max(array)
    for i, element in enumerate(array):
        array[i] = (element - minimum) / (maximum-minimum)

    return array


def evaluate(layer_sizes, num_cars, num_episodes, learn_rate, discount_factor):
    nets = [NeuralNetwork(layer_sizes) for _ in range(num_cars)]

    pygame.font.init()
    font = pygame.font.SysFont('timesnewroman', 30)
    clock = pygame.time.Clock()

    for episode in range(num_episodes):

        cars = [Car() for _ in range(num_cars)]
        training_data = []
        dead_cars = []
        rewards = []
        next_iteration = False
        timer = 0

        while len(dead_cars) < len(cars):
            timer += .1
            if (timer >= 130):
                break
            dt = clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    next_iteration = True

            if next_iteration:
                break
            
            SCREEN.blit(TRACK, (0,0))
            SCREEN.blit(font.render(f"episode {episode}, time elapsed: {int(timer)}", False, (255, 255, 255)), (0,0))

            dead_cars = []

            for i, car in enumerate(cars):
                inputs = car.update(dt, clock)
                if not car.alive:
                    dead_cars.append(i)
                    car.alive = False
                    continue
                # The forward pass
                output = nets[i].forward_pass(inputs)
                decision = np.argmax(output)

                if decision == 0:
                    # left
                    car.direction = -1
                elif decision == 2:
                    # right
                    car.direction = 1
                else:
                    # straight
                    car.direction = 0

                training_data.append(DataPoint(inputs))

                pygame.sprite.GroupSingle(car).draw(SCREEN)
            pygame.display.update()

        # episode_costs = [net.cost(training_data, output) for net in nets]
        # print(f"cost of nets for episode {episode}", [cost for cost in episode_costs])

        for idx, net in enumerate(nets):
            # backward pass
            net.learn(training_data, learn_rate)
            training_data.clear()



if __name__ == '__main__':
    layer_sizes = [3, 6, 3]
    num_cars = 1
    num_episodes = 1000
    learn_rate = .1
    discount_factor = 0.1
    evaluate(layer_sizes, num_cars, num_episodes, learn_rate, discount_factor)