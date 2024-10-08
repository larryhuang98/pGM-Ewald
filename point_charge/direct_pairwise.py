import xml.etree.ElementTree as ET
import numpy as np
from scipy.special import erfc
from itertools import product

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

def calculate_correction_term(coordinates, charges, box):
    L = np.mean(box)
    correction = 0
    total_dipole_moment = np.zeros(3)  # Sum of q * r
    for i in range(len(charges)):
        total_dipole_moment += charges[i] * coordinates[i]
    
    correction = np.dot(total_dipole_moment, total_dipole_moment)
    return 2 * np.pi / (3 * L**3) * correction

def calculate_energy(coordinates, charges, box, spherical_size):
    N = len(coordinates)
    L = np.array(box)
    
    # Generate lattice vectors for periodic images
    m_range = range(-spherical_size, spherical_size + 1)
    lattice_vectors = np.array(list(product(m_range, m_range, m_range)))
    
    energy = 0.0
    for i in range(N):
        for j in range(N):
            ri = coordinates[i]
            rj = coordinates[j]
            qi = charges[i]
            qj = charges[j]
            
            for m in lattice_vectors:
                if i == j and np.all(m == 0):
                    continue
                
                r = ri - rj + m * L
                r_norm = np.linalg.norm(r)
                energy += 0.5 * qi * qj / r_norm
    return energy

def main():
    xml_file = "system_info.xml"
    xyz_file = "coordinates.xyz"
    
    box, particles = read_xml(xml_file)
    coordinates = read_xyz(xyz_file)
    charges = np.array([p['charge'] for p in particles])
    
    max_spherical_size = 10
    for size in range(max_spherical_size + 1):
        energy = calculate_energy(coordinates, charges, box, size)
        print(f"Spherical size: {size}, Energy: {energy:.8f}")
    
    correction = calculate_correction_term(coordinates, charges, box)
    print(f"Final total energy: {energy-correction:.8f}")

if __name__ == "__main__":
    main()
