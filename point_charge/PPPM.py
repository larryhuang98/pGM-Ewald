import xml.etree.ElementTree as ET
import numpy as np
from scipy.special import erfc

def read_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    box = [float(root.find(f'box/{dim}').text) for dim in ['length', 'width', 'height']]
    particles = []
    for particle in root.findall('particles/particle'):
        name = particle.find('name').text
        charge = float(particle.find('charge').text)
        particles.append({'name': name, 'charge': charge})
    
    return box, particles

def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    coordinates = []
    for line in lines[2:]:
        _, x, y, z = line.split()
        coordinates.append([float(x), float(y), float(z)])
    
    return np.array(coordinates)