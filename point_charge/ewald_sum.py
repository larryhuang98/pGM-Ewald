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

def minimum_image_distance(r_i, r_j, box):
    d = r_i - r_j
    d -= box * np.round(d / box)
    return d, np.linalg.norm(d)

def ewald_summation(coordinates, charges, box, kappa, k_max, r_cut):
    N = len(charges)
    V = np.prod(box)
    
    def real_space_sum():
        energy = 0.0
        for i in range(N):
            for j in range(i+1, N):
                r_ij, r = minimum_image_distance(coordinates[i], coordinates[j], box)
                if r < r_cut:
                    energy += charges[i] * charges[j] * erfc(kappa * r) / r
        return energy

    def reciprocal_space_sum():
        energy = 0.0
        k_vectors = []
        for nx in range(-k_max, k_max+1):
            for ny in range(-k_max, k_max+1):
                for nz in range(-k_max, k_max+1):
                    if nx == ny == nz == 0:
                        continue
                    k = 2 * np.pi * np.array([nx/box[0], ny/box[1], nz/box[2]])
                    k_vectors.append(k)
        
        for k in k_vectors:
            k_squared = np.dot(k, k)
            structure_factor = np.sum(charges * np.exp(1j * np.dot(coordinates, k)))
            energy += (4*np.pi / V) * (np.exp(-k_squared / (4*kappa**2)) / k_squared) * np.abs(structure_factor)**2
        return energy.real / 2

    def self_interaction():
        return -kappa / np.sqrt(np.pi) * np.sum(charges**2)

    def dipole_correction():
        dipole = np.sum(charges[:, np.newaxis] * coordinates, axis=0)
        return 2 * np.pi / (3 * V) * np.dot(dipole, dipole)

    energy_real = real_space_sum()
    energy_reciprocal = reciprocal_space_sum()
    energy_self = self_interaction()
    energy_dipole = dipole_correction()
    
    total_energy = energy_real + energy_reciprocal + energy_self + energy_dipole
    return total_energy, energy_dipole

def main():
    xml_file = "system_info.xml"
    xyz_file = "coordinates.xyz"
    
    box, particles = read_xml(xml_file)
    coordinates = read_xyz(xyz_file)
    charges = np.array([p['charge'] for p in particles])

    kappa = 0.5  # Ewald convergence parameter
    k_max = 10  # Maximum wave vector for reciprocal sum
    r_cut = min(box) / 2  # Set cutoff to half the smallest box dimension

    energy, dipole_correction = ewald_summation(coordinates, charges, np.array(box), kappa, k_max, r_cut)
    
    print(f"Ewald sum energy: {energy:.8f}")

if __name__ == "__main__":
    main()