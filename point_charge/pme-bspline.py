import xml.etree.ElementTree as ET
import numpy as np
from scipy.special import erfc, erf
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

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
    d -= np.round(d / box) * box
    return d, np.linalg.norm(d)

def plot_charge_distribution(Q, grid_size):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.meshgrid(np.arange(grid_size[0]),
                          np.arange(grid_size[1]),
                          np.arange(grid_size[2]))

    x, y, z = x.flatten(), y.flatten(), z.flatten()
    Q_flat = Q.flatten()

    sizes = np.abs(Q_flat) / np.max(np.abs(Q_flat)) * 100
    ax.scatter(x[Q_flat > 0], y[Q_flat > 0], z[Q_flat > 0], c='red', s=sizes[Q_flat > 0], alpha=0.6, label='Positive')
    ax.scatter(x[Q_flat < 0], y[Q_flat < 0], z[Q_flat < 0], c='blue', s=sizes[Q_flat < 0], alpha=0.6, label='Negative')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Charge Distribution on Grid')
    ax.legend()

    plt.tight_layout()
    plt.savefig("charge.png")


def M_n(u, n):
    if n == 2:
        return max(0, 1 - abs(u - 1)) if 0 <= u <= 2 else 0
    else:
        return (u / (n - 1)) * M_n(u, n - 1) + ((n - u) / (n - 1)) * M_n(u - 1, n - 1)

# Function to distribute charges on the grid using B-splines
def distribute_charges(coordinates, charges, box, grid_size, order):
    Q = np.zeros(grid_size, dtype=float)
    h = box / grid_size

    for r, q in zip(coordinates, charges):
        u = r / h
        for n1 in range(order):
            for n2 in range(order):
                for n3 in range(order):
                    k1 = (int(u[0]) + n1) % grid_size[0]
                    k2 = (int(u[1]) + n2) % grid_size[1]
                    k3 = (int(u[2]) + n3) % grid_size[2]
                    
                    M1 = M_n(u[0] - (k1 - int(u[0])), order)
                    M2 = M_n(u[1] - (k2 - int(u[1])), order)
                    M3 = M_n(u[2] - (k3 - int(u[2])), order)
                    
                    Q[k1, k2, k3] += q * M1 * M2 * M3

    return Q

# Function to compute the B-spline factors for reciprocal space calculations
def b_spline_factor(m, order, grid_size):
    b = np.ones(3, dtype=complex)
    for d in range(3):
        if 2 * abs(m[d]) == grid_size[d] and order % 2 == 1:
            b[d] = 0
        else:
            exp_term = np.exp(1j * np.pi * m[d] * (order - 1) / grid_size[d])
            sum_term = sum(M_n(k + 1, order) * np.exp(2j * np.pi * m[d] * k / grid_size[d]) for k in range(order))
            
            b[d] = exp_term / sum_term if abs(sum_term) > 1e-10 else 0

    return b

# Function to compute the reciprocal space contribution to PME energy
def pme_reciprocal(coordinates, charges, box, beta, grid_size, order):
    V = np.prod(box)
    Q = distribute_charges(coordinates, charges, box, grid_size, order)
    F_Q = fftn(Q)
    
    energy = 0.0
    for m1 in range(grid_size[0]):
        for m2 in range(grid_size[1]):
            for m3 in range(grid_size[2]):
                if m1 == m2 == m3 == 0:
                    continue
                m = np.array([m1, m2, m3])
                m_adj = np.where(m > grid_size // 2, m - grid_size, m)
                m_sq = np.sum((2 * np.pi * m_adj / box)**2)
                
                if m_sq == 0:
                    continue
                
                b = b_spline_factor(m, order, grid_size)
                B_m = np.abs(b[0])**2 * np.abs(b[1])**2 * np.abs(b[2])**2
                exp_factor = np.exp(-np.pi**2 * m_sq / beta**2) / m_sq
                energy += exp_factor * B_m * np.abs(F_Q[m1, m2, m3])**2
    
    energy *= 1 / (2 * np.pi * V)
    return energy.real

# Real space calculation: E = 1/2 Sum qi * qj erfc(beta |rj-ri+n|)/|rj-ri+n|
def pme_real(coordinates, charges, box, beta, r_cut):
    energy = 0
    N = len(charges)
    for i in range(N):
        for j in range(i + 1, N):
            r_ij, r = minimum_image_distance(coordinates[i], coordinates[j], box)
            if r < r_cut:
                energy += charges[i] * charges[j] * erfc(beta * r) / r
    return energy


# PME correction term calculation: E = -1/2 Sum qi qj erf(beta |ri -rj|) -beta/Sqrt(Pi) Sum qi^2
def pme_correction(coordinates, charges, box, beta, r_cut):
    N = len(charges)
    energy = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm((coordinates[i] - coordinates[j]))
            energy -=  charges[i] * charges[j] * erf(beta * r) / r
    energy -= beta / np.sqrt(np.pi) * np.sum(charges**2)
    return energy


def pme(coordinates, charges, box, beta, grid_size, order, r_cut):
    coordinates = coordinates % box
    energy_real = pme_real(coordinates, charges, box, beta, r_cut)
    energy_reciprocal = pme_reciprocal(coordinates, charges, box, beta, grid_size, order)
    energy_correction = pme_correction(coordinates, charges, box, beta, r_cut)
    
    total_energy = energy_real + energy_reciprocal + energy_correction
    return total_energy

def main():
    xml_file = "system_info.xml"
    xyz_file = "coordinates.xyz"
    
    box, particles = read_xml(xml_file)
    coordinates = read_xyz(xyz_file)
    charges = np.array([p['charge'] for p in particles])

    kappa = 0.5
    grid_size = np.array([8, 8, 8])
    order = 6
    r_cut = min(box) / 2

    Q = distribute_charges(coordinates, charges, box, grid_size, order)
    plot_charge_distribution(Q, grid_size)

    energy = pme(coordinates, charges, box, kappa, grid_size, order, r_cut)
    print(f"PME total energy: {energy:.6f}")

if __name__ == "__main__":
    main()
