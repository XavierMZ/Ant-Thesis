#!/usr/bin/env python

from pygame.math import Vector2 as Vector
import numpy as np
import math
import random
import copy
import json


## Function to Find Neighbords of a Grid Square

def get_grid_neighbors(x,y,w,h):
    neighbors = []
    for x2 in range(x-1, x+2):
        for y2 in range(y-1, y+2):
            if (
                -1 < x <= w-1 and -1 < y <= h-1 
                and (x != x2 or y != y2) 
                and (0 <= x2 <= w-1) and (0 <= y2 <= h-1)
               ):
                neighbors.append((x2,y2))
    return neighbors


## Ant Class

class ANT:
    def __init__(self, x, y, num, IIC, IANT_RADIUS, IMIN_SPEED, IMAX_SPEED, IK_P, IK_S,
                 IP_FORAGING, IP_HOMING, IMIN_PH_SENS, ITHETA_STOCHASTICITY, IVMAX, IIPS):
        # ANT CONSTANTS
        self.ANT_RADIUS = IANT_RADIUS
        self.MIN_SPEED = IMIN_SPEED
        self.MAX_SPEED = IMAX_SPEED
        self.K_P = IK_P
        self.K_S = IK_S
        self.P_FORAGING = IP_FORAGING
        self.P_HOMING = 1 - IP_FORAGING
        self.MIN_PH_SENS = IMIN_PH_SENS
        self.THETA_STOCHASTICITY = ITHETA_STOCHASTICITY
        self.VMAX = IVMAX
        self.IPS = IIPS
        
        clustered = int(IIC == "c")
        self.num = num
        self.pos = Vector(x,y)
        self.theta = random.uniform(0,2*np.pi)
        self.speed = self.MIN_SPEED + (1-clustered)*random.uniform(0,self.MAX_SPEED - self.MIN_SPEED)
        if random.uniform(0,1) <= self.P_FORAGING:
            self.mode = "FORAGING"
        else:
            self.mode = "HOMING"
        
    def avoid_wall(self,w,h):
        if round(self.pos.x) <= (1 + self.ANT_RADIUS) or round(self.pos.x) >= (w - 1 - self.ANT_RADIUS)             or int(self.pos.y) <= (self.ANT_RADIUS + 1) or round(self.pos.y) >= (h - 1 - self.ANT_RADIUS):
            center = (round(random.uniform(0.1*w,0.9*w)),round(random.uniform(0.1*h,0.9*h)))
            direction = (center[0] - round(self.pos.x), center[1] - round(self.pos.y))
            self.theta = math.atan2(direction[1], direction[0])
    
    def avoid_collision(self, ants):
        xdot = self.speed/self.IPS * np.cos(self.theta)
        ydot = self.speed/self.IPS * np.sin(self.theta)

        new_x_coord = round(self.pos.x + xdot)
        new_y_coord = round(self.pos.y + ydot)
        for ant in ants:
            if abs(new_x_coord - round(ant.pos.x)) == 0 and abs(new_y_coord - round(ant.pos.y)) == 0:
                self.theta += random.uniform(-0.5*self.THETA_STOCHASTICITY,0.5*self.THETA_STOCHASTICITY)
                return False
        return True
    
    # Basic function to find gradient by looking for neighbor with highest concentration of pheromone.
    def find_gradient_direction_1(self,pheromone_field):
        w, h = len(pheromone_field[0]), len(pheromone_field)
        x, y = round(self.pos.x), round(self.pos.y)
        x, y = min(max(x,0),w), min(max(y,0),h)
        neighbors = get_grid_neighbors(x,y,w,h)
        maximum_pheromone = self.MIN_PH_SENS * self.VMAX #potential_field[location_x][location_y]
        max_pheromone_neighbors = [(x,y)]
        for n in neighbors:
            n_x, n_y = n
            if pheromone_field[n_x][n_y] > 1.1 * maximum_pheromone:
                maximum_pheromone = pheromone_field[n_x][n_y]
                max_pheromone_neighbors = [(n_x,n_y)]
            elif pheromone_field[n_x][n_y] > 0.9 * maximum_pheromone:
                max_pheromone_neighbors.append((n_x,n_y))
        max_pheromone_neighbor = random.choice(max_pheromone_neighbors)

        return max_pheromone_neighbor, maximum_pheromone
    
    def move_ant(self, pheromone_field, ants):
        w, h = len(pheromone_field[0]), len(pheromone_field)
        if self.mode == "FORAGING":
            self.theta += random.uniform(-self.THETA_STOCHASTICITY,self.THETA_STOCHASTICITY)
            self.avoid_wall(w,h)
            
            self.speed = max(self.speed, (self.MIN_SPEED + self.MAX_SPEED) / 2)
            
            xdot = self.speed/self.IPS * np.cos(self.theta)
            ydot = self.speed/self.IPS * np.sin(self.theta)
            
            move = self.avoid_collision(ants)
            if move:
                self.pos.x += xdot
                self.pos.y += ydot
        
        elif self.mode == "HOMING":
            gradient_direction, gradient_V = self.find_gradient_direction_1(pheromone_field)
            gradient = (gradient_direction[0] - round(self.pos.x), gradient_direction[1] - round(self.pos.y))
            if gradient[0] != 0:
                gradient_theta = math.atan2(gradient[1], gradient[0])
            elif gradient[1] == 0:
                gradient_theta = self.theta + random.uniform(-self.THETA_STOCHASTICITY,self.THETA_STOCHASTICITY)
            else:
                gradient_theta = (np.pi/2)*(gradient[1] - round(self.pos.y))

            thetadot = self.K_P*np.sin(gradient_theta - self.theta)
            self.theta += thetadot

            self.avoid_wall(w,h)
            
            # slow down or speed up
            bounded_gradient_V = min(max(0,gradient_V),self.VMAX)
            target_speed = self.MAX_SPEED * (1 - bounded_gradient_V / self.VMAX)
            self.speed = self.speed + self.K_S * (target_speed - self.speed)
            self.speed = min(max(self.MIN_SPEED,self.speed),self.MAX_SPEED)
            
            xdot = self.speed/self.IPS * np.cos(self.theta)
            ydot = self.speed/self.IPS * np.sin(self.theta)
            
            move = self.avoid_collision(ants)
            if move:
                self.pos.x += xdot
                self.pos.y += ydot


## Simulation Class


class SIM:
    def __init__(self, DIMS, NUM_ANTS, IIPS, IIC,
                 ID, IE, IVMIN, IVMAX, IM, IP_DROP, IPHEROMONE_INTERVAL,
                 IANT_RADIUS, IMIN_SPEED, IMAX_SPEED, IK_P, IK_S,
                 IP_FORAGING, IP_HOMING, IMIN_PH_SENS, ITHETA_STOCHASTICITY):
        self.width, self.height = DIMS
        
        # PHEROMONE CONSTANTS
        self.D = ID #0.2
        self.E = IE #0.001
        self.VMIN = IVMIN
        self.VMAX = IVMAX
        self.M = IM
        self.P_DROP = IP_DROP
        self.PHEROMONE_INTERVAL = IPHEROMONE_INTERVAL
        self.IPS = IIPS
        
        self.pheromone_field = [[0 for i in range(self.width+1)] for j in range(self.height+1)]
        
        # ANTS
        clustered = int(IIC == "c")
        a_per_line = math.floor(math.sqrt(NUM_ANTS))
        a_spacing = clustered*2 + (1 - clustered)*math.floor(self.width / (a_per_line+1))
        X_OFFSET = 0 + clustered*round(self.width/2 - a_spacing*(a_per_line/2) - 1)
        Y_OFFSET = 0 + clustered*round(self.height/2 - a_spacing*(a_per_line/2) - 1)
        self.ants = [ANT(x * a_spacing + X_OFFSET, y * a_spacing + Y_OFFSET, y + (x-1)*a_per_line, IIC,
                         IANT_RADIUS, IMIN_SPEED, IMAX_SPEED, IK_P, IK_S,
                         IP_FORAGING, 1 - IP_FORAGING, IMIN_PH_SENS, 
                         ITHETA_STOCHASTICITY, IVMAX, IIPS) for x in range(1,a_per_line+1) for y in range(1,a_per_line+1)]
        
        # If ants start clustered, add pheromone around their initial conditions.
        if clustered:
            for i in range(X_OFFSET+1,X_OFFSET+a_per_line*a_spacing+1,2):
                for j in range(Y_OFFSET+1,Y_OFFSET+a_per_line*a_spacing+1,2):
                    self.pheromone_field[i][j] += self.VMAX

        # TIMER
        self.transpired_intervals = 0
    
    def update(self):
        self.transpired_intervals += 1
        self.evolve_pheromone()
        self.drop_pheromone()
        for ant in self.ants:
            ant.move_ant(self.pheromone_field, self.ants)
    
    def drop_pheromone(self):
        for ant in self.ants:
            if random.uniform(0,1) < self.P_DROP/self.IPS:
                # Determine ant coordinates.
                drop_x, drop_y = round(ant.pos.x), round(ant.pos.y)
                # Bound coordinates to within simulation dimensions.
                drop_x, drop_y = max(min(drop_x,self.width-1),1), max(min(drop_y,self.height-1),1)
                # Add drop.
                self.pheromone_field[drop_x][drop_y] = self.VMAX * self.M
    
    def evolve_pheromone(self):
        p_width = len(self.pheromone_field[0])
        p_height = len(self.pheromone_field)
    
        old_field = copy.deepcopy(self.pheromone_field)
        for x in range(1,p_width-1):
            for y in range(1,p_height-1):
                square_concentration = old_field[x][y]
                neighbors = get_grid_neighbors(x,y,p_width,p_height)
                total_pheromone = square_concentration
                for n in neighbors:
                    total_pheromone += old_field[n[0]][n[1]]
                ave_pheromone = total_pheromone / (len(neighbors) + 1)
                next_concentration = (1 - self.E/self.IPS) * (square_concentration + (self.D/self.IPS * (ave_pheromone - square_concentration)))
                self.pheromone_field[x][y] = next_concentration


## Function to Run Simulation without Visualization


def run_simulation(IDURATION, IIPS, IWIDTH, IHEIGHT, IN_ANTS, IIC,
                   ID, IE, IVMIN, IVMAX, IM, IP_DROP, IPHEROMONE_INTERVAL, 
                   IANT_RADIUS, IMIN_SPEED, IMAX_SPEED, IK_P, IK_S, 
                   IP_FORAGING, IP_HOMING, IMIN_PH_SENS, ITHETA_STOCHASTICITY):
    # SIMULATION CONSTANTS
    DURATION = IDURATION
    IPS = IIPS

    # initialize simulation
    simulation = SIM((IWIDTH,IHEIGHT), IN_ANTS, IIPS, IIC,
                     ID, IE, IVMIN, IVMAX, IM, IP_DROP, IPHEROMONE_INTERVAL,
                     IANT_RADIUS, IMIN_SPEED, IMAX_SPEED, IK_P, IK_S,
                     IP_FORAGING, IP_HOMING, IMIN_PH_SENS, ITHETA_STOCHASTICITY)

    # List to save all ant locations.
    ant_locations = []

    # Loop to run simulation.
    while simulation.transpired_intervals <= DURATION*IPS:
        current_ant_locations = []
        for ant in simulation.ants:
            current_ant_locations.append((ant.pos.x,ant.pos.y))
        ant_locations.append(current_ant_locations)

        simulation.update()
    
    return ant_locations


## Parameter Analysis Data Collection:

# SIMULATION CONSTANTS
iDURATION = 10
iIPS = 5
iWIDTH = 100
iHEIGHT = 100
iN_ANTS = 1
iIC = "nc"
# PHEROMONE CONSTANTS
iD = [0.1,0.2,0.3,0.4,0.5] #0.2     # percentage of pheromone in a square that diffuses per second
iE = [0.001,0.005,0.01] #0.001      # percentage of pheromone in a square that evaporates per second
iVMIN = 0
iVMAX = 100
iM = 10
iP_DROP = 1     # per second
iPHEROMONE_INTERVAL = 1
# ANT CONSTANTS
iANT_RADIUS = 1
iMIN_SPEED = 2  # squares per second
iMAX_SPEED = 10 # squares per second
iK_P = 0.75
iK_S = 0.5
iP_FORAGING = 0.0
iP_HOMING = 1 - iP_FORAGING
iMIN_PH_SENS = 0.2
iTHETA_STOCHASTICITY = np.pi/8

reps = 1
trials = {}
print("Completed")
for d in iD:
    print("D:", d)
    p2trials = {}
    for e in iE:
        print("\tE:", e)
        p2reps = {}
        for i in range(reps):
            print("\t\tRep", i + 1)
            ant_trajectories = run_simulation(iDURATION, iIPS, iWIDTH, iHEIGHT, iN_ANTS, iIC,
                                              d, e, 
                                              iVMIN, iVMAX, iM, iP_DROP, iPHEROMONE_INTERVAL, 
                                              iANT_RADIUS, iMIN_SPEED, iMAX_SPEED, iK_P, iK_S, 
                                              iP_FORAGING, iP_HOMING, iMIN_PH_SENS, iTHETA_STOCHASTICITY)
            p2reps["r"+str(i+1)] = ant_trajectories
        p2trials[e] = p2reps
    trials[d] = p2trials

output_file = "data/new_data_file.txt"
with open(output, "w") as f:
    f.write(json.dumps(trials))