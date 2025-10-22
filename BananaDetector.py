#Aida Asmardokht

import torch
import math
import matplotlib.pyplot as plt

print("Designing a detector for banana cargo at port")
print("="*55)

print("\n" + "="*20)
print("A- calculation of neutrino flux ")
print("="*20)

class BananaNeutrinoCalculator:
    def __init__(self):
        self.potassium_per_banana = torch.tensor(0.0005)    # 0.5g K per banana
        self.K40_fraction = torch.tensor(0.000117)          # 0.0117% K-40
        self.K40_half_life = torch.tensor(1.248e9 * 365 * 24 * 3600)  # seconds

    def calculate_flux(self, num_bananas, distance, offloading_time=3*24*3600):
       # Calculation of neutrino flux during 3 days of offloading
       # Calculation of radioactive activity
       avogadro = 6.022e23
       molar_mass_K = 0.040
        
       total_potassium = num_bananas * self.potassium_per_banana
       K40_atoms = total_potassium * self.K40_fraction * avogadro / molar_mass_K
        
       decay_constant = torch.log(torch.tensor(2.0)) / self.K40_half_life
       total_activity = decay_constant * K40_atoms
       # All decays of potassium-40 produce neutrinos (100%)
       total_neutrinos_per_sec = total_activity * 1.0

       # Constant flux over 3 days
       constant_flux = total_neutrinos_per_sec / (4 * math.pi * distance**2)

       return {
            'total_activity': total_activity.item(),
            'constant_flux': constant_flux.item(),
            'total_neutrinos_per_sec': total_neutrinos_per_sec.item(),
            'total_neutrinos_3days': total_neutrinos_per_sec.item() * 3 * 24 * 3600
        }

# Calculation for a typical ship
calculator = BananaNeutrinoCalculator()
num_bananas = torch.tensor(20000000)  #20 million bananas
distance = torch.tensor(100.0)        # 100 meters distance

flux_result = calculator.calculate_flux(num_bananas, distance)

print(f" Ship parameters:")
print(f"   • Number of bananas: {num_bananas.item():,}")
print(f"   • Detector distance: {distance.item()} meters")
print(f"   • Docking time: 3 full days (no offloading)")

print(f"\n Neutrino flux calculation results (constant over 3 days):")
print(f"   • Radioactive activity: {flux_result['total_activity']:.2e} Bq")
print(f"   • Neutrinos per second: {flux_result['total_neutrinos_per_sec']:.2e} s^(-1)")
print(f"   • Constant neutrino flux: {flux_result['constant_flux']:.1f} (m²/s)^(-1)")
print(f"   • Total neutrinos in 3 days: {flux_result['total_neutrinos_3days']:.2e}")

##########################################################################################
########################################################################################
print("\n" + "="*20)
print("B- possible reaction channels to detect the flux")
print("="*20)



class WorkingDetectionChannels:
    def __init__(self, flux, detector_mass=1000, exposure_time=3*24*3600):
        self.flux = torch.tensor(flux)
        self.detector_mass = torch.tensor(detector_mass)
        self.exposure_time = torch.tensor(exposure_time)
        
    def calculate_events(self, cross_section, target_density):
        """Calculating Expected Events"""
        events = self.flux * target_density * cross_section * self.exposure_time
        return events.item()
    
    def analyze_channels(self):
       #Analysis of detection channels
        molar_mass_water = 0.018  # kg/mol
        avogadro = 6.022e23
        
        water_molecules = self.detector_mass / molar_mass_water * avogadro
        
        
        print(f"  Water molecules: {water_molecules:.2e}")
        
        # Calculating targets       
        electrons = water_molecules * 10
        protons = water_molecules * 10
        neutrons = water_molecules * 8
        nuclei = water_molecules * 3
        
        print(f"    Electrons: {electrons:.2e}")
        print(f"    Protons: {protons:.2e}")
        print(f"    neutrons: {neutrons:.2e}")
        print(f"    nuclei : {nuclei:.2e}")
        print()

        # Cross sections (m²)
        cross_sections = {
            'Elastic Scattering (ν + e⁻)': 1e-45,
            'Coherent Scattering (ν + nucleus)': 1e-42,
            'Charged Current (νₑ + n → e⁻ + p)': 1e-47,
            'Neutral Current (ν + n → ν + n)': 1e-46,
            'Inverse Beta Decay (ν̄ₑ + p → e⁺ + n)': 1e-43
        }
        
        # Target for each channel
        targets = {
            'Elastic Scattering (ν + e⁻)': electrons,
            'Coherent Scattering (ν + nucleus)': nuclei,
            'Charged Current (νₑ + n → e⁻ + p)': neutrons,
            'Neutral Current (ν + n → ν + n)': neutrons,
            'Inverse Beta Decay (ν̄ₑ + p → e⁺ + n)': protons
        }
        
    
        
        results = {}
        for channel, sigma in cross_sections.items():
            target_density = targets[channel]
            events = self.calculate_events(sigma, target_density)
            results[channel] = events
            
            status = "DETECTABLE" if events >= 1 else "NOT DETECTABLE"
            print(f"{status} {channel}:")
            print(f"    events: {events:.2e}")
            print(f"    sigma: {sigma:.1e} m²")
            print()

        
        
        return results
detector = WorkingDetectionChannels(flux=250000)
detector.analyze_channels()

##################################################################################
##################################################################################
print("\n" + "="*20)
print("C- why the inverse beta decay is not a possible solution")
print("="*20)

class IBDAnalysis:
    def __init__(self):
        self.ibd_threshold = torch.tensor(1.806)  # MeV
        self.banana_neutrino_energy = torch.tensor(1.31)  # MeV
        
    def analyze_ibd(self):
    #IBD Possibility Analysis
    
        
        print(f"IBD energy threshold: {self.ibd_threshold.item()} MeV")
        print(f" Banana neutrino energy: {self.banana_neutrino_energy.item()} MeV")
        
        energy_deficit = self.ibd_threshold - self.banana_neutrino_energy
        print(f"Energy deficit: {energy_deficit.item():.3f} MeV")        
 # Check if IBD is possible
        if self.banana_neutrino_energy < self.ibd_threshold:
            fraction_above_threshold = torch.tensor(0.0)
            print(f"Fraction of neutrinos above threshold: {fraction_above_threshold.item():.3f}")
            
            print("\n IBD IS NOT POSSIBLE")
            print("Reasons:")
            print("- Banana neutrino energy (1.31 MeV) < IBD threshold (1.806 MeV)")
            print("- Energy deficit of 0.496 MeV cannot be overcome")
            print("- 0% of banana neutrinos have sufficient energy for IBD")
            
        else:
            fraction_above_threshold = (self.banana_neutrino_energy - self.ibd_threshold) / self.banana_neutrino_energy
            print(f"Fraction of neutrinos above threshold: {fraction_above_threshold.item():.3f}")
            print(" IBD IS POSSIBLE")

        print("\n Conclusion: Inverse Beta Decay cannot be used for")
        print("banana neutrino detection due to insufficient energy.")
        
        return fraction_above_threshold.item()
   
   
   
    


ibd_analyzer = IBDAnalysis()

ibd_analyzer.analyze_ibd()
