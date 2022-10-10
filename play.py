import pygame
from environment import Environment


env = Environment()
time_step = env.reset()

while True:
    # get key inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    action = [0,0]
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action[0] = -.5
    if keys[pygame.K_RIGHT]:
        action[0] = .5
    if keys[pygame.K_UP]:
        action[1] = +.5
    if keys[pygame.K_DOWN]:
        action[1] = -.5
    
    time_step = env.step(action)
    env.render(telemetry=True)
